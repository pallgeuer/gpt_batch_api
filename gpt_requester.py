# OpenAI GPT requester

# Imports
from __future__ import annotations
import os
import json
import logging
import contextlib
import dataclasses
from typing import Optional, Union, Self, Any, ContextManager, Iterable
import httpx
import openai
from . import utils
from .logger import log

# Constants
DEFAULT_ENDPOINT = '/v1/chat/completions'

# Logging configuration
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

#
# Common types
#

# GPT request class
@dataclasses.dataclass(frozen=True)
class GPTRequest:
	payload: dict[str, Any]                # Request payload (JSON-compatible OpenAI request) => Is batched into JSONL files and uploaded to OpenAI for processing
	endpoint: Optional[str] = None         # Endpoint to use for the request (e.g. /v1/chat/completions, None = Default)
	meta: Optional[dict[str, Any]] = None  # Optional custom metadata to associate with this request (JSON-compatible) => Is batched and stored in the state file while OpenAI is processing the payloads, and is returned alongside the response

#
# State file
#

# Queue state class
@dataclasses.dataclass
class QueueState:
	max_request_id: int = 0                                                                         # Maximum request ID used thus far (0 = No request ID used so far)
	request_id_meta: dict[int, Optional[dict[str, Any]]] = dataclasses.field(default_factory=dict)  # A map of request ID to custom metadata for all requests in the request queue (NOT including the request pool, ordered by ascending request ID)

# Batch state class
@dataclasses.dataclass
class BatchState:
	request_id_meta: dict[int, Optional[dict[str, Any]]]  # A map of request ID to custom metadata for all requests in the batch (order is the same as in the local JSONL batch input file, which is by ascending request ID)
	local_jsonl: str                                      # Path of the generated local JSONL batch input file (requests are ordered by ascending request ID)
	local_jsonl_size: int                                 # Number of bytes in the local JSONL batch input file
	remote_jsonl_id: Optional[str] = None                 # Assigned string ID of the uploaded batch input file (OpenAI file storage)
	remote_batch: Optional[str] = None                    # Assigned string ID of the running batch (OpenAI batches)
	# TODO: Need like num_requests, num_tokens, etc? Whatever is relevant for cutoffs/thresholds/decisions
	# TODO: Always assert sizes when creating request_id_meta dicts (BatchState) to make sure no key overlap

# State class
@dataclasses.dataclass
class State:
	version: int = 1                                                     # Data format version number
	queue: QueueState = dataclasses.field(default_factory=QueueState)    # State of the request queue
	batches: list[BatchState] = dataclasses.field(default_factory=list)  # States of all batches that are currently pending or in progress

# State file class
class StateFile:

	state: Optional[State]

	def __init__(self, path: str):
		# path = Path to the JSON state file to load/save/manage (nominally *.json extension)
		self.path = os.path.abspath(path)
		log.info(f"GPT requester state file: {self.path}")
		self.name = os.path.basename(self.path)
		self._enter_stack = contextlib.ExitStack()
		self.state = None

	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.callback(self.unload)
			try:
				self.load()
			except FileNotFoundError:
				self.create()
			assert self.state is not None
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self):
		self.state = State()
		self.save(stack=None)

	def load(self):
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.state = utils.dataclass_from_json(cls=State, json_data=file)
		log.info(f"Loaded GPT requester state file with {len(self.state.queue.request_id_meta)} queued requests and {len(self.state.batches)} batches ({utils.format_size(file_size)}): {self.name}")

	def save(self, stack: Optional[contextlib.ExitStack[Optional[bool]]]):
		with utils.SafeOpenForWrite(self.path, stack=stack) as file:
			utils.json_from_dataclass(obj=self.state, file=file)
			file_size = utils.get_file_size(file)
		log.info(f"Saved GPT requester state file with {len(self.state.queue.request_id_meta)} queued requests and {len(self.state.batches)} batches ({utils.format_size(file_size)}): {self.name}")

	def unload(self):
		self.state = None

#
# Queue file
#

# GPT request item class (each line of the JSONL queue file corresponds to one of these items)
@dataclasses.dataclass(frozen=True)
class GPTRequestItem:
	id: int          # Unique monotonic ID corresponding to the GPT request
	req: GPTRequest  # The actual GPT request
	endpoint: str    # Resolved endpoint to use for the request

# GPT request information class
@dataclasses.dataclass(frozen=True)
class GPTRequestInfo:
	json: str        # Full batch-style request in compact JSON string form (includes the payload, always ends with a single trailing newline)
	json_size: int   # Number of bytes in the compact JSON string form when encoded with UTF-8
	num_tokens: int  # A conservative approximation (overestimation) of how many input tokens the request payload has (the aim is to never actually end up with more input tokens in a batch than the sum of these approximations suggests)

# Cached GPT request class
@dataclasses.dataclass(frozen=True)
class CachedGPTRequest:

	item: GPTRequestItem  # GPT request item (backed by the queue file)
	info: GPTRequestInfo  # GPT request information (automatically calculated from the GPT request item, and cached in memory)

	@staticmethod
	def from_item(item: GPTRequestItem) -> CachedGPTRequest:
		full_request = dict(custom_id=f'id-{item.id}', method='POST', url=item.endpoint, body=item.req.payload)
		compact_json = json.dumps(full_request) + '\n'
		return CachedGPTRequest(
			item=item,
			info=GPTRequestInfo(
				json=compact_json,
				json_size=len(compact_json.encode('utf-8')),
				num_tokens=estimate_payload_tokens(item.req.payload),
			),
		)

# Pool/queue class
@dataclasses.dataclass(frozen=True)
class PoolQueue:

	pool: list[CachedGPTRequest] = dataclasses.field(default_factory=list)             # Request pool: Requests added to the GPT requester are first added to the request pool, which is purely in memory and not backed by a file (the request pool is 'lost' if a KeyboardInterrupt or crash occurs)
	queue: list[Optional[CachedGPTRequest]] = dataclasses.field(default_factory=list)  # Request queue: When the GPT requester is made to commit, all requests in the request pool are moved to the request queue, which is backed by the queue file and thus persistent

	@property
	def queue_len(self) -> int:
		return sum(1 for cached_req in self.queue if cached_req is not None)

	def queue_request_id_meta(self) -> dict[int, Optional[dict[str, Any]]]:
		request_id_meta = {cached_req.item.id: cached_req.item.req.meta for cached_req in self.queue if cached_req is not None}
		if len(request_id_meta) != self.queue_len:
			raise ValueError("Request queue contains duplicate request IDs")
		return request_id_meta

# Queue file class
class QueueFile:

	pool_queue: Optional[PoolQueue]

	def __init__(self, path: str):
		# path = Path to the JSONL queue file to load/save/manage (nominally *.jsonl extension)
		self.path = os.path.abspath(path)
		log.info(f"GPT requester queue file: {self.path}")
		self.name = os.path.basename(self.path)
		self._enter_stack = contextlib.ExitStack()
		self.pool_queue = None

	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.callback(self.unload)
			try:
				self.load()
			except FileNotFoundError:
				self.create()
			assert self.pool_queue is not None
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self):
		self.pool_queue = PoolQueue()
		self.save(stack=None)

	def load(self):
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.pool_queue = PoolQueue(pool=[], queue=[CachedGPTRequest.from_item(utils.dataclass_from_json(cls=GPTRequestItem, json_data=line)) for line in file.readlines()])
		log.info(f"Loaded GPT requester queue file with {self.pool_queue.queue_len} requests ({utils.format_size(file_size)}): {self.name}")

	def save(self, stack: Optional[contextlib.ExitStack[Optional[bool]]]):
		with utils.SafeOpenForWrite(self.path, stack=stack) as file:
			for cached_req in self.pool_queue.queue:
				if cached_req is not None:
					utils.json_from_dataclass(obj=cached_req.item, file=file)
					file.write('\n')
			file_size = utils.get_file_size(file)
		log.info(f"Saved GPT requester queue file with {self.pool_queue.queue_len} requests ({utils.format_size(file_size)}): {self.name}")

	def unload(self):
		self.pool_queue = None

#
# GPT requester
#

# GPT requester class
class GPTRequester:

	# Construct a GPT requester to make use of the OpenAI Batch API to process GPT requests
	def __init__(
		self,
		working_dir: str,                                     # Path to the GPT working directory to use (will be used for automatically managed lock, state, requests and batch files)
		autocreate_working_dir: bool = True,                  # Whether to automatically create the GPT working directory if it does not exist (parent directory must already exist)
		name_prefix: str = 'gpt_requester',                   # Name prefix to use for all files created in the GPT working directory
		lock_timeout: Optional[float] = None,                 # Timeout (if any) to use when attempting to lock exclusive access to the files in the GPT working directory corresponding to the given name prefix (see utils.LockFile)
		lock_poll_interval: Optional[float] = None,           # Lock file polling interval (see utils.LockFile)
		lock_status_interval: Optional[float] = None,         # Lock file status update interval (see utils.LockFile)
		openai_api_key: Optional[str] = None,                 # OpenAI API key (see openai.OpenAI, ends up in request headers)
		openai_organization: Optional[str] = None,            # OpenAI organization (see openai.OpenAI, ends up in request headers)
		openai_project: Optional[str] = None,                 # OpenAI project (see openai.OpenAI, ends up in request headers)
		client_base_url: Union[str, httpx.URL, None] = None,  # Base URL to use for the OpenAI API client (see openai.OpenAI, servers other than OpenAI's servers can be configured to expose an OpenAI API with suitable endpoints)
		client_kwargs: Optional[dict[str, Any]] = None,       # Additional kwargs to use for the OpenAI API client (see openai.OpenAI)
		client: Optional[openai.OpenAI] = None,               # OpenAI API client instance to use, if given, otherwise one is created internally (Note: If an explicit client instance is given, the preceding client_* and openai_* arguments are ignored)
		default_endpoint: Optional[str] = None,               # Default endpoint to use for GPT requests that don't explicitly specify one (None = OPENAI_ENDPOINT environment variable with the DEFAULT_ENDPOINT constant as a fallback)
	):

		self.working_dir = os.path.abspath(working_dir)
		working_dir_parent = os.path.dirname(self.working_dir)
		if not os.path.isdir(working_dir_parent):
			raise ValueError(f"Parent directory of GPT working directory needs to exist: {working_dir_parent}")

		self.name_prefix = name_prefix
		if not self.name_prefix:
			raise ValueError("Name prefix cannot be empty")

		self.autocreate_working_dir = autocreate_working_dir
		created_working_dir = False
		if self.autocreate_working_dir:
			with contextlib.suppress(FileExistsError):
				os.mkdir(self.working_dir)
				created_working_dir = True

		log.info(f"Using GPT requester of prefix '{self.name_prefix}' in dir: {self.working_dir}{' [CREATED]' if created_working_dir else ''}")
		if not os.path.isdir(self.working_dir):
			raise FileNotFoundError(f"GPT working directory does not exist: {self.working_dir}")

		self.lock = utils.LockFile(path=os.path.join(self.working_dir, f"{self.name_prefix}.lock"), timeout=lock_timeout, poll_interval=lock_poll_interval, status_interval=lock_status_interval)
		self.state = StateFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_state.json"))
		self.queue = QueueFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_queue.jsonl"))

		self.client = client or openai.OpenAI(api_key=openai_api_key, organization=openai_organization, project=openai_project, base_url=client_base_url, **client_kwargs)
		self.default_endpoint = default_endpoint
		if self.default_endpoint is None:
			self.default_endpoint = os.environ.get("OPENAI_ENDPOINT")
		if self.default_endpoint is None:
			self.default_endpoint = DEFAULT_ENDPOINT

		self._enter_stack = contextlib.ExitStack()
		self.S: Optional[State] = None
		self.PQ: Optional[PoolQueue] = None
		self.P: Optional[list[CachedGPTRequest]] = None
		self.Q: Optional[list[Optional[CachedGPTRequest]]] = None
		self.max_request_id: Optional[int] = None

	# Enter method for the required use of GPTRequester as a context manager (acquires a lock on the state files and such and reads them in)
	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.enter_context(self.lock)
			stack.callback(self.on_exit)
			stack.enter_context(self.state)
			self.S = self.state.state
			self.max_request_id = self.S.queue.max_request_id
			stack.enter_context(self.queue)
			self.PQ = self.queue.pool_queue
			self.P = self.PQ.pool
			self.Q = self.PQ.queue
			self.validate_state_queue(clean=True)
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	# Exit method for the required use of GPTRequester as a context manager (saves and releases all state files and such)
	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		if self.P:
			log.warning(f"Exiting GPT requester with {len(self.P)} uncommitted requests in the request pool")
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	# Local actions to perform on exit
	def on_exit(self):
		self.S = self.PQ = self.P = self.Q = None
		self.max_request_id = None

	# Validate that there are no obvious issues with the current state and queue (clean refers to the expected status right after a commit)
	def validate_state_queue(self, *, clean: bool):

		if clean:
			if self.P:
				raise ValueError("Request pool state is unexpectedly not clean")
			if any(cached_req is None for cached_req in self.Q):
				raise ValueError("Request queue state is unexpectedly not clean")
			if self.S.queue.max_request_id != self.max_request_id:
				raise ValueError(f"Maximum request ID is inconsistent between GPT requester and queue state: {self.max_request_id} vs {self.S.queue.max_request_id}")

		if self.S.queue.request_id_meta != self.PQ.queue_request_id_meta():
			raise ValueError("ID-meta map is inconsistent between the state and queue files for the request queue")

		for batch in self.S.batches:
			if not utils.is_ascending(batch.request_id_meta.keys(), strict=True):
				raise ValueError(f"The request IDs in a batch are not strictly ascending: {list(batch.request_id_meta.keys())}")
		if not utils.is_ascending(self.S.queue.request_id_meta.keys(), strict=True):
			raise ValueError(f"The request IDs in the request queue are not strictly ascending: {list(self.S.queue.request_id_meta.keys())}")
		if not utils.is_ascending((pool_request_ids := [cached_req.item.id for cached_req in self.P]), strict=True):
			raise ValueError(f"The request IDs in the request pool are not strictly ascending: {pool_request_ids}")

		req_id_intervals = [(min(req_ids), max(req_ids)) for req_ids in (*(batch.request_id_meta.keys() for batch in self.S.batches), self.S.queue.request_id_meta.keys(), pool_request_ids)]
		if not utils.is_ascending((req_id for req_id_interval in req_id_intervals for req_id in req_id_interval), strict=True):
			raise ValueError(f"The request ID intervals overlap between the batches, request queue and request pool: {req_id_intervals}")

	# Add a single request to the request pool without committing it to the request queue just yet (the request will be committed in the next call to commit_requests())
	def add_request(self, req: GPTRequest):
		self.max_request_id += 1
		item = GPTRequestItem(id=self.max_request_id, req=req, endpoint=req.endpoint if req.endpoint is not None else self.default_endpoint)
		self.P.append(CachedGPTRequest.from_item(item))

	# Add multiple requests to the request pool without committing them to the request queue just yet (the requests will be committed in the next call to commit_requests())
	# Note that reqs can also for convenience be a generator that executes some loop over source samples and yields instances of GPTRequest
	def add_requests(self, reqs: Iterable[GPTRequest]):
		for req in reqs:
			self.add_request(req)

	# Optionally add some request(s) to the request pool, then in any case commit all requests now in the pool to the request queue, and then optionally batch the requests as much as possible (use force_batch=True to ensure all requests are batched), and push as many batches as possible for remote processing
	# The body of the context manager is executed within an AtomicExitStack (an ExitStack that is not interruptible by a keyboard interrupt) that must completely reverse all actions taken if an exception is raised at any point whatsoever during the entered stack
	# The idea is that the body of the entered commit_requests() context manager should be used to save to disk whatever state is necessary so that all requests added to the GPT requester so far do not get generated again in this or a new run (as they are now committed, i.e. locked in), even if the current run were to be keyboard interrupted (or crash)
	@contextlib.contextmanager
	def commit_requests(self, reqs: Union[Iterable[GPTRequest], GPTRequest, None] = None, batch: bool = True, force_batch: bool = False, push: bool = True) -> ContextManager[contextlib.ExitStack[Optional[bool]]]:

		if reqs is not None:
			if isinstance(reqs, GPTRequest):
				self.add_request(reqs)
			else:
				self.add_requests(reqs)

		def revert_queue(P: list[CachedGPTRequest], Q: list[Optional[CachedGPTRequest]]):
			self.P[:] = P
			self.Q[:] = Q

		def revert_queue_state(queue_state_dict: dict[str, Any]):
			for field, value in queue_state_dict.items():
				setattr(self.S.queue, field, value)

		with utils.AtomicExitStack() as stack:
			if self.P or any(cached_req is None for cached_req in self.Q):
				stack.callback(revert_queue, P=self.P.copy(), Q=self.Q.copy())
				self.Q[:] = [cached_req for cached_req in self.Q if cached_req is not None]
				self.Q.extend(self.P)
				self.P.clear()
				stack.callback(revert_queue_state, queue_state_dict=dataclasses.asdict(self.S.queue))  # noqa
				self.S.queue.max_request_id = self.max_request_id
				self.S.queue.request_meta_id = self.PQ.queue_request_id_meta()
				self.validate_state_queue(clean=True)
				self.queue.save(stack=stack)
				self.state.save(stack=stack)
			else:
				self.validate_state_queue(clean=True)
			yield stack  # Reaching this yield signifies that all requests that have been added to the request pool have been successfully committed to the request queue, queue file and state (BUT if the code executed during the yield raises an exception then ALL of this will be perfectly reversed)

		if batch:
			self.batch_requests(force=force_batch, push=push)

	# Process the request queue and generate full batches as much as possible (also generate a non-full trailing batch with whatever requests are left if force=True), and then optionally push the batches as much as possible for remote processing
	def batch_requests(self, force: bool = False, push: bool = True):
		# TODO: Implement (also trailing batch if force=True)
		# TODO: Auto-monitor when you have enough requests to fill a batch (or multiple) and send them off
		if push:
			self.push_requests()

	# Push as many batches as possible for remote processing
	def push_requests(self):
		pass  # TODO: See what unpushed batches there are and push them

	# TODO: User test case is one that attempts to get character codes and descriptions of unicode characters (tests that UTF-8 and everything is working correctly and the characters arrive properly on OpenAI servers)
	# TODO: Options like 'model' should be in the user state file (to remain consistent)

	# TODO: Methods to wait for batches (including timeout, maybe print how long waiting etc), and process their results and return it to user code
	# TODO: Safety margin that adds safety margin to all thresholds/limits (e.g. makes the limits 90% of the values actually passed)
	# TODO: 100MB or 100MiB limit?

	# TODO: OPENAI_BASE_URL is used in the client with a fallback of f"https://api.openai.com/v1" as the base URL -> When making isolated single requests, retrieve the client base_url and add the endpoint on the end, omitting a URL path component if one ends with the same one another one starts with? OR something to a similar effect?

	# TODO: Count tokens etc from single requests in a separate counter, and add them to a combined total elsewhere
	# TODO: When creating a batch, add metadata about the GPT requester prefix, batch ID (monotonic int), local machine, local batch path, PID, etc... so that it can be uniquely identified where/why the batch comes from (e.g. dict(host=os.uname().nodename, script=__file__, action='annotate_batch', batch_local=local_json_file, batch_remote=remote_json_file.id))

	# TODO: Auto-compute stats about the request (monotonic ID must be assigned, state file with next monotonic ID is only updated when request is committed!), convert it to string, etc -> But autocomputed stats don't necessarily need to be saved to file, e.g. the string conversion makes no sense to save => GPTRequestInfo(frozen)?
	# TODO: Explicit UTF-8 encoding EVERYWHERE anything is saved! ALSO newline='' => BATCH API requires no-BOM UTF-8 with Unix endings (and no empty lines of course)

	# TODO: If there is a standalone action with no synchronicity required with any other action, then being atomic is enough
	# TODO: Otherwise, everything that requires state synchronicity should be linked inside an exitstack that undoes EVERYTHING perfectly if anything fails (this could mean keeping a backup copy of a state file before it was successfully changed, in case a later synchronised action errors)
	# TODO: For convenience, such exit stacks should delay KeyboardInterrupt to avoid completely unnecessary reversions based on a badly timed user Ctrl+C
	# TODO: Summary: Either single atomic operation or multiple perfectly reversible ones in utils.DelayKeyboardInterruptStack

	# TODO: How to gracefully deal with an entire batch erroring out? Raise a runtime exception as no clue whether data is the problem or wrong limits configured, but not auto-recoverable in any case? Or maybe after 2-3 failed batches raise? Failed push? Failed result?
	# TODO: Make direct (non-batched) immediate/sync request (sync_request(GPTRequest) -> GPTResponse)

	# TODO: Debug mode that does everything (including uploading files?) except ACTUALLY start batches (everything that doesn't cost money)
	# TODO: Maybe have two bools for no_upload / no_cost
	# TODO: Function to direct evaluate a single request without using the Batch API (single_request() -> Response)
	# TODO: Metrics (tokens in/out (what happens about thought tokens), per batch, num requests, divided by successful/unsuccessful)
	# TODO: Wandb (the entire 'metrics' part of the current state, plus how many batches are active etc)
	# TODO: Remove tqdm if it is not actually used (would be better for wandb logs)
	# TODO: Any meaningful way to implement auto-retry individual failed requests? (up to a certain retry count)
	# TODO: Also need a feature to NOT keep retrying requests/samples indefinitely that are just failing (hard because they get reconstructed or might be batched where only 1 of 15 is the failing reason)

#
# Miscellaneous
#

# Conservatively approximate (overestimate) how many input tokens a request payload has (the aim is to never actually end up with more input tokens in a batch than the sum of these approximations suggests)
def estimate_payload_tokens(payload: dict[str, Any]) -> int:
	raise NotImplementedError  # TODO: Parse /v1/chat/completions, /v1/embeddings, and /v1/completions -style requests for text/images/tokens. SPECIAL IMAGES handling (low res vs high res)!
# EOF
