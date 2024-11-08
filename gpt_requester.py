# OpenAI GPT requester

# Imports
from __future__ import annotations
import os
import json
import time
import inspect
import logging
import argparse
import datetime
import itertools
import contextlib
import dataclasses
from typing import Optional, Union, Self, Any, ContextManager, Iterable
import httpx
import openai
from .logger import log
from . import tokens, utils

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

# Remote batch state class
@dataclasses.dataclass
class RemoteBatchState:
	file_id: str                # Assigned string ID of the uploaded batch input file (OpenAI file storage)
	batch_id: str               # Assigned string ID of the running batch (OpenAI batches)
	status: openai.types.Batch  # Batch status as per the last time the server was checked (this is not updated anymore once the batch status is for the first time in one of the finished states)
	finished: bool = False      # Whether the batch is known to have finished on the remote server by now (status as per the last time the server was checked, updated whenever status is updated)

# Batch state class
@dataclasses.dataclass
class BatchState:
	id: int = 0                                                                                     # Unique monotonic ID corresponding to the batch
	local_jsonl: str = ''                                                                           # Path of the generated local JSONL batch input file (requests are ordered by ascending request ID)
	local_jsonl_size: int = 0                                                                       # Number of bytes in the local JSONL batch input file
	num_requests: int = 0                                                                           # Number of requests in the batch
	num_tokens: dict[int, int] = dataclasses.field(default_factory=dict)                            # An approximation of the number of input tokens in each request payload
	total_tokens: int = 0                                                                           # An approximation of the total number of input tokens in all of the request payloads together
	request_id_meta: dict[int, Optional[dict[str, Any]]] = dataclasses.field(default_factory=dict)  # A map of request ID to custom metadata for all requests in the batch (order is the same as in the local JSONL batch input file, which is by ascending request ID)
	full_batch: bool = False                                                                        # Whether this is a full batch, i.e. some threshold was reached that triggered this batch to be formed, as opposed to the batch being forced
	remote: Optional[RemoteBatchState] = None                                                       # Remote state of the batch once it has been pushed

# State class
@dataclasses.dataclass
class State:
	version: int = 1                                                     # Data format version number
	queue: QueueState = dataclasses.field(default_factory=QueueState)    # State of the request queue
	max_batch_id: int = 0                                                # Maximum batch ID used thus far (0 = No batch ID used so far)
	batches: list[BatchState] = dataclasses.field(default_factory=list)  # States of all batches that are currently pending or in progress (in ascending batch ID order)

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
		with self._enter_stack as stack, utils.AtomicExitStack() as atomic_stack:
			stack.callback(self.unload)
			try:
				self.load()
			except FileNotFoundError:
				self.create(stack=atomic_stack)
			assert self.state is not None
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self, stack: contextlib.ExitStack[Optional[bool]]):
		self.state = State()
		self.save(stack=stack)

	def load(self):
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.state = utils.dataclass_from_json(cls=State, json_data=file)
		log.info(f"Loaded GPT requester state file with {len(self.state.queue.request_id_meta)} queued requests and {len(self.state.batches)} batches ({utils.format_size(file_size)}): {self.name}")

	def save(self, stack: contextlib.ExitStack[Optional[bool]]):
		with utils.SafeOpenForWrite(path=self.path, stack=stack) as file:
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
	num_tokens: int  # An approximation of how many input tokens the request payload has

# Cached GPT request class
@dataclasses.dataclass(frozen=True)
class CachedGPTRequest:

	item: GPTRequestItem  # GPT request item (backed by the queue file)
	info: GPTRequestInfo  # GPT request information (automatically calculated from the GPT request item, and cached in memory)

	@staticmethod
	def from_item(item: GPTRequestItem, token_estimator: tokens.TokenEstimator) -> CachedGPTRequest:
		full_request = dict(custom_id=f'id-{item.id}', method='POST', url=item.endpoint, body=item.req.payload)
		compact_json = json.dumps(full_request, ensure_ascii=False, indent=None) + '\n'
		return CachedGPTRequest(
			item=item,
			info=GPTRequestInfo(
				json=compact_json,
				json_size=len(compact_json.encode('utf-8')),
				num_tokens=token_estimator.payload_input_tokens(payload=item.req.payload, endpoint=item.endpoint).total,
			),
		)

# Pool/queue class
@dataclasses.dataclass(frozen=True)
class PoolQueue:

	pool: list[CachedGPTRequest] = dataclasses.field(default_factory=list)             # Request pool: Requests added to the GPT requester are first added to the request pool, which is purely in memory and not backed by a file (the request pool is 'lost' if a KeyboardInterrupt or crash occurs)
	queue: list[Optional[CachedGPTRequest]] = dataclasses.field(default_factory=list)  # Request queue: When the GPT requester is made to commit, all requests in the request pool are moved to the request queue, which is backed by the queue file and thus persistent

	@property
	def pool_len(self) -> int:
		return len(self.pool)

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

	def __init__(self, path: str, token_estimator: tokens.TokenEstimator):
		# path = Path to the JSONL queue file to load/save/manage (nominally *.jsonl extension)
		# token_estimator = Token estimator instance to use
		self.path = os.path.abspath(path)
		log.info(f"GPT requester queue file: {self.path}")
		self.name = os.path.basename(self.path)
		self.token_estimator = token_estimator
		self._enter_stack = contextlib.ExitStack()
		self.pool_queue = None

	def __enter__(self) -> Self:
		with self._enter_stack as stack, utils.AtomicExitStack() as atomic_stack:
			stack.callback(self.unload)
			try:
				self.load()
			except FileNotFoundError:
				self.create(atomic_stack)
			assert self.pool_queue is not None
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self, stack: contextlib.ExitStack[Optional[bool]]):
		self.pool_queue = PoolQueue()
		self.save(stack=stack)

	def load(self):
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.pool_queue = PoolQueue(pool=[], queue=[CachedGPTRequest.from_item(
				item=utils.dataclass_from_json(cls=GPTRequestItem, json_data=line),
				token_estimator=self.token_estimator,
			) for line in file.readlines()])
		log.info(f"Loaded GPT requester queue file with {self.pool_queue.queue_len} requests ({utils.format_size(file_size)}): {self.name}")

	def save(self, stack: contextlib.ExitStack[Optional[bool]]):
		with utils.SafeOpenForWrite(path=self.path, stack=stack) as file:
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
		name_prefix: str,                                     # Name prefix to use for all files created in the GPT working directory (e.g. 'my_gpt_requester')
		*,                                                    # Keyword arguments only beyond here

		openai_api_key: Optional[str] = None,                 # OpenAI API key (see openai.OpenAI, ends up in request headers)
		openai_organization: Optional[str] = None,            # OpenAI organization (see openai.OpenAI, ends up in request headers)
		openai_project: Optional[str] = None,                 # OpenAI project (see openai.OpenAI, ends up in request headers)
		client_base_url: Union[str, httpx.URL, None] = None,  # Base URL to use for the OpenAI API client (see openai.OpenAI, servers other than OpenAI's servers can be configured to expose an OpenAI API with suitable endpoints)
		client_kwargs: Optional[dict[str, Any]] = None,       # Additional kwargs to use for the OpenAI API client (see openai.OpenAI)
		client: Optional[openai.OpenAI] = None,               # OpenAI API client instance to use, if given, otherwise one is created internally (Note: If an explicit client instance is given, the preceding client_* and openai_* arguments are ignored)
		default_endpoint: Optional[str] = None,               # Default endpoint to use for GPT requests that don't explicitly specify one (None = OPENAI_ENDPOINT environment variable with the DEFAULT_ENDPOINT constant as a fallback)

		autocreate_working_dir: bool = True,                  # Whether to automatically create the GPT working directory if it does not exist (parent directory must already exist)
		lock_timeout: Optional[float] = None,                 # Timeout (if any) to use when attempting to lock exclusive access to the files in the GPT working directory corresponding to the given name prefix (see utils.LockFile)
		lock_poll_interval: Optional[float] = None,           # Lock file polling interval (see utils.LockFile)
		lock_status_interval: Optional[float] = None,         # Lock file status update interval (see utils.LockFile)
		token_estimator_warn: str = 'once',                   # Warning mode to use for internal token estimator (see tokens.TokenEstimator)
		remote_update_interval: float = 10.0,                 # Interval in multiples of which to update remote batch states when waiting for remote batches to finish (seconds)

		# TODO: min_batch_requests => Less than this causes direct API to be used instead for the requests that would normally have ended up in the batch
		max_batch_requests: int = 50000,                      # Maximum number of requests allowed in a batch
		max_batch_mb: int = 100,                              # Maximum allowed batch size in MB (not MiB)
		max_batch_ktokens: int = 2000,                        # Maximum number of tokens to include in a single batch (in units of 1000)
		max_unpushed_batches: int = 10,                       # Maximum number of unpushed local batches at any one time
		max_remote_batches: int = 100,                        # Maximum number of remote batches at any one time
		max_remote_requests: int = 5000000,                   # Maximum number of requests across all uploaded remote batches at any one time
		max_remote_mb: int = 10000,                           # Maximum allowed total size in MB (not MiB) of all uploaded remote batches at any one time
		max_remote_ktokens: int = 5000,                       # Maximum allowed total number of tokens (in units of 1000) across all uploaded remote batches at any one time
		max_token_safety: float = 1.05,                       # Safety factor to use when comparing token counts to specified maximum values (token counts are ultimately approximations until the batch is actually executed, so a safety factor can be useful in ensuring that token limits are truly never exceeded in practice)
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

		self.token_estimator = tokens.TokenEstimator(warn=token_estimator_warn)
		self.lock = utils.LockFile(path=os.path.join(self.working_dir, f"{self.name_prefix}.lock"), timeout=lock_timeout, poll_interval=lock_poll_interval, status_interval=lock_status_interval)
		self.state = StateFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_state.json"))
		self.queue = QueueFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_queue.jsonl"), token_estimator=self.token_estimator)

		self.client = client or openai.OpenAI(api_key=openai_api_key, organization=openai_organization, project=openai_project, base_url=client_base_url, **client_kwargs)
		self.default_endpoint = default_endpoint
		if self.default_endpoint is None:
			self.default_endpoint = os.environ.get("OPENAI_ENDPOINT")
		if self.default_endpoint is None:
			self.default_endpoint = DEFAULT_ENDPOINT

		self.remote_update_interval = remote_update_interval
		if self.remote_update_interval < 1.0:
			raise ValueError(f"Remote batch update interval must be at least 1.0s: {self.remote_update_interval}")

		self.max_batch_requests = max_batch_requests
		if self.max_batch_requests < 1:
			raise ValueError(f"Maximum number of requests in a batch must be at least 1: {self.max_batch_requests}")
		self.max_batch_mb = max_batch_mb
		if self.max_batch_mb < 1:
			raise ValueError(f"Maximum batch size in MB must be at least 1: {self.max_batch_mb}")
		self.max_batch_ktokens = max_batch_ktokens
		if self.max_batch_ktokens < 1:
			raise ValueError(f"Maximum batch ktokens must be at least 1: {self.max_batch_ktokens}")
		self.max_unpushed_batches = max_unpushed_batches
		if self.max_unpushed_batches < 1:
			raise ValueError(f"Maximum number of unpushed batches must be at least 1: {self.max_unpushed_batches}")
		self.max_remote_batches = max_remote_batches
		if self.max_remote_batches < 1:
			raise ValueError(f"Maximum number of remote batches must be at least 1: {self.max_remote_batches}")
		self.max_remote_requests = max_remote_requests
		if self.max_remote_requests < self.max_batch_requests:
			raise ValueError(f"Maximum number of requests across all remote batches must be at least as great as the maximum number of requests in a batch: {self.max_remote_requests} vs {self.max_batch_requests}")
		self.max_remote_mb = max_remote_mb
		if self.max_remote_mb < self.max_batch_mb:
			raise ValueError(f"Maximum total uploaded remote batch size in MB must be at least as great as the maximum batch size in MB: {self.max_remote_mb} vs {self.max_batch_mb}")
		self.max_remote_ktokens = max_remote_ktokens
		if self.max_remote_ktokens < self.max_batch_ktokens:
			raise ValueError(f"Maximum total uploaded remote batch ktokens must be at least as great as the maximum batch ktokens: {self.max_remote_ktokens} vs {self.max_batch_ktokens}")
		self.max_token_safety = max_token_safety
		if self.max_token_safety < 1.0:
			raise ValueError(f"The number of tokens safety factor must be at least 1.0: {self.max_token_safety}")

		self._enter_stack = contextlib.ExitStack()
		self.S: Optional[State] = None
		self.PQ: Optional[PoolQueue] = None
		self.P: Optional[list[CachedGPTRequest]] = None
		self.Q: Optional[list[Optional[CachedGPTRequest]]] = None
		self.max_request_id: Optional[int] = None

	# Dictionary of all __init__ keyword arguments and their default values
	__kwargs__ = {name: param.default for name, param in inspect.signature(__init__).parameters.items() if name != 'self' and param.default != inspect.Parameter.empty}

	# Populate a dictionary of __init__ keyword arguments based on an attribute-based config object (e.g. argparse.Namespace, flat omegaconf.DictConfig)
	@classmethod
	def get_kwargs(cls, cfg: utils.Config) -> dict[str, Any]:
		return {key: getattr(cfg, key) for key in cls.__kwargs__.keys() if hasattr(cfg, key)}

	# Configure an argparse parser to incorporate an argument group for the keyword arguments that can be passed to the init of this class
	@classmethod
	def configure_argparse(cls, parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup], *, title: Optional[str] = 'GPT requester', description: Optional[str] = None, group_kwargs: Optional[dict[str, Any]] = None, **custom_defaults) -> argparse._ArgumentGroup:  # noqa

		if isinstance(parser, argparse.ArgumentParser):
			group = parser.add_argument_group(title=title, description=description, **(group_kwargs if group_kwargs is not None else {}))
		else:
			group = parser

		# noinspection PyShadowingBuiltins
		def add_argument(name: str, type: type, help: str, unit: str = ''):
			default_value = custom_defaults.get(name, cls.__kwargs__[name])
			group.add_argument(f'--{name}', type=type, default=default_value, help=help if default_value is None else f'{help} [default: {default_value}{unit}]')

		add_argument(name='openai_api_key', type=str, help="OpenAI API key (see openai.OpenAI, ends up in request headers)")
		add_argument(name='openai_organization', type=str, help="OpenAI organization (see openai.OpenAI, ends up in request headers)")
		add_argument(name='openai_project', type=str, help="OpenAI project (see openai.OpenAI, ends up in request headers)")
		add_argument(name='client_base_url', type=str, help="Base URL to use for the OpenAI API client (see openai.OpenAI, servers other than OpenAI's servers can be configured to expose an OpenAI API with suitable endpoints)")
		add_argument(name='default_endpoint', type=str, help="Default API endpoint to use")

		add_argument(name='autocreate_working_dir', type=bool, help="Whether to automatically create the GPT working directory if it does not exist (parent directory must already exist)")
		add_argument(name='lock_timeout', type=float, unit='s', help="Timeout (if any) to use when attempting to lock exclusive access to the files in the GPT working directory corresponding to the given name prefix (see utils.LockFile)")
		add_argument(name='lock_poll_interval', type=float, unit='s', help="Lock file polling interval (see utils.LockFile)")
		add_argument(name='lock_status_interval', type=float, unit='s', help="Lock file status update interval (see utils.LockFile)")
		add_argument(name='token_estimator_warn', type=str, help="Warning mode to use for internal token estimator (see tokens.TokenEstimator)")
		add_argument(name='remote_update_interval', type=float, unit='s', help="Interval in multiples of which to update remote batch states when waiting for remote batches to finish")

		add_argument(name='max_batch_requests', type=int, help="Maximum number of requests allowed in a batch")
		add_argument(name='max_batch_mb', type=int, unit='MB', help="Maximum allowed batch size in MB (not MiB)")
		add_argument(name='max_batch_ktokens', type=int, unit=' ktokens', help="Maximum number of tokens to include in a single batch (in units of 1000)")
		add_argument(name='max_unpushed_batches', type=int, help="Maximum number of unpushed local batches at any one time")
		add_argument(name='max_remote_batches', type=int, help="Maximum number of remote batches at any one time")
		add_argument(name='max_remote_mb', type=int, unit='MB', help="Maximum allowed total size in MB (not MiB) of all uploaded remote batches at any one time")
		add_argument(name='max_remote_ktokens', type=int, unit=' ktokens', help="Maximum allowed total number of tokens (in units of 1000) across all uploaded remote batches at any one time")
		add_argument(name='max_token_safety', type=int, help="Safety factor to use when comparing token counts to specified maximum values (token counts are ultimately approximations until the batch is actually executed, so a safety factor can be useful in ensuring that token limits are truly never exceeded in practice)")

		return group

	# Enter method for the required use of GPTRequester as a context manager
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

	# Exit method for the required use of GPTRequester as a context manager
	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		if self.P:
			log.warning(f"{self.name_prefix}: Exiting GPT requester with {len(self.P)} uncommitted requests in the request pool")
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
			if batch.id <= 0:
				raise ValueError(f"Invalid batch ID: {batch.id}")
			if batch.id > self.S.max_batch_id:
				raise ValueError(f"Batch ID is greater than the supposed maximum assigned batch ID: {batch.id} > {self.S.max_batch_id}")
			if batch.remote is not None:
				if batch.remote.file_id != batch.remote.status.input_file_id:
					raise ValueError(f"Batch file ID is not consistent: {batch.remote.file_id} vs {batch.remote.status.input_file_id}")
				if batch.remote.batch_id != batch.remote.status.id:
					raise ValueError(f"Batch ID is not consistent: {batch.remote.batch_id} vs {batch.remote.status.id}")
		if not utils.is_ascending((batch.id for batch in self.S.batches), strict=True):
			raise ValueError(f"The batch IDs are not strictly ascending: {list(batch.id for batch in self.S.batches)}")

		for batch in self.S.batches:
			if not utils.is_ascending(batch.request_id_meta.keys(), strict=True):
				raise ValueError(f"The request IDs in a batch are not strictly ascending: {list(batch.request_id_meta.keys())}")
		if not utils.is_ascending(self.S.queue.request_id_meta.keys(), strict=True):
			raise ValueError(f"The request IDs in the request queue are not strictly ascending: {list(self.S.queue.request_id_meta.keys())}")
		if not utils.is_ascending((pool_request_ids := [cached_req.item.id for cached_req in self.P]), strict=True):
			raise ValueError(f"The request IDs in the request pool are not strictly ascending: {pool_request_ids}")

		req_id_intervals = [(min(req_ids, default=None), max(req_ids, default=None)) for req_ids in (*(batch.request_id_meta.keys() for batch in self.S.batches), self.S.queue.request_id_meta.keys(), pool_request_ids)]
		flat_req_id_bounds = [req_id for req_id_interval in req_id_intervals for req_id in req_id_interval if req_id is not None]
		if not utils.is_ascending(flat_req_id_bounds, strict=True):
			raise ValueError(f"The request ID intervals overlap between the batches, request queue and request pool: {req_id_intervals}")
		if any(req_id > self.S.queue.max_request_id for req_id in flat_req_id_bounds):
			raise ValueError(f"There are request ID(s) greater than the supposed maximum assigned request ID {self.S.queue.max_request_id}: {req_id_intervals}")

	# TODO: direct_request() (+comment) => Go through add_request => commit_requests => batch => push => process cycle and pretend everything is immediate, and make exactly all those changes (e.g. max_request_id incremented, save state (careful as don't actually literally want to save Nones and stuff, but wait, we have no reason to touch the queue file anyway right?), etc)

	# Add a single request to the request pool without committing it to the request queue just yet (the request will be committed in the next call to commit_requests())
	def add_request(self, req: GPTRequest):
		self.max_request_id += 1
		item = GPTRequestItem(id=self.max_request_id, req=req, endpoint=req.endpoint if req.endpoint is not None else self.default_endpoint)
		self.P.append(CachedGPTRequest.from_item(item=item, token_estimator=self.token_estimator))

	# Add multiple requests to the request pool without committing them to the request queue just yet (the requests will be committed in the next call to commit_requests())
	# Note that reqs can also for convenience be a generator that executes some loop over source samples and yields instances of GPTRequest
	def add_requests(self, reqs: Iterable[GPTRequest]):
		for req in reqs:
			self.add_request(req)

	# Optionally add some request(s) to the request pool, then in any case commit all requests now in the pool to the request queue
	# The body of the context manager is executed within an AtomicExitStack (an ExitStack that is not interruptible by a keyboard interrupt) that must completely reverse all actions taken if an exception is raised at any point whatsoever during the entered stack
	# The idea is that the body of the entered commit_requests() context manager should be used to save to disk whatever state is necessary so that all requests added to the GPT requester so far (given by a list[CachedGPTRequest]) do not get generated again in this or a new run (as they are now committed, i.e. locked in), even if the current run were to be immediately keyboard interrupted (or crash)
	@contextlib.contextmanager
	def commit_requests(self, reqs: Union[Iterable[GPTRequest], GPTRequest, None] = None) -> ContextManager[tuple[list[CachedGPTRequest], contextlib.ExitStack[Optional[bool]]]]:

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
			cached_reqs = self.P.copy()
			if self.P or any(cached_req is None for cached_req in self.Q):
				stack.callback(revert_queue, P=cached_reqs, Q=self.Q.copy())
				self.Q[:] = [cached_req for cached_req in self.Q if cached_req is not None]
				self.Q.extend(self.P)
				self.P.clear()
				stack.callback(revert_queue_state, queue_state_dict=dataclasses.asdict(self.S.queue))  # noqa
				self.S.queue.max_request_id = self.max_request_id
				self.S.queue.request_id_meta = self.PQ.queue_request_id_meta()
				self.validate_state_queue(clean=True)
				self.queue.save(stack=stack)
				self.state.save(stack=stack)
			else:
				self.validate_state_queue(clean=True)
			yield cached_reqs, stack  # Reaching this yield signifies that all requests that have been added to the request pool have been successfully committed to the request queue, queue file and state (BUT if the code executed during the yield raises an exception then ALL of this will be perfectly reversed)
			stack.pop_all()

	# Process the request queue and generate full batches as much as possible (also generate a trailing non-full batch with whatever requests are left if force=True)
	def batch_requests(self, force: bool = False) -> int:

		num_created = 0
		num_created_requests = 0

		index = 0
		while index < len(self.Q):

			num_unpushed_batches = self.num_unpushed_batches()
			if num_unpushed_batches >= self.max_unpushed_batches:
				log.info(f"{self.name_prefix}: Not batching {sum(1 for cached_req in itertools.islice(self.Q, index, None) if cached_req is not None)} requests left in the request queue as there are already {num_unpushed_batches} unpushed batches (max {self.max_unpushed_batches} allowed)")
				break

			batch_index = index
			batch_reqs = []
			batch = BatchState()

			num_nones = 0
			while index < len(self.Q):

				cached_req = self.Q[index]
				if cached_req is None:
					num_nones += 1
				else:

					next_local_jsonl_size = batch.local_jsonl_size + cached_req.info.json_size
					next_num_requests = batch.num_requests + 1
					next_total_tokens = batch.total_tokens + cached_req.info.num_tokens

					if (
						next_num_requests >= self.max_batch_requests or
						next_local_jsonl_size >= self.max_batch_mb * 1000000 or
						next_total_tokens * self.max_token_safety >= self.max_batch_ktokens * 1000
					):
						if batch.num_requests <= 0:
							raise ValueError(f"The batch limits are too strict to allow even a single request to be added: Batch requests {next_num_requests} >= {self.max_batch_requests} OR Batch bytes {next_local_jsonl_size} >= {self.max_batch_mb * 1000000} OR Batch tokens {next_total_tokens} >= {round(self.max_batch_ktokens * 1000 / self.max_token_safety)}")
						batch.full_batch = True
						break

					req_id = cached_req.item.id
					batch_reqs.append(cached_req.info.json)
					batch.local_jsonl_size = next_local_jsonl_size
					batch.num_requests = next_num_requests
					assert req_id not in batch.num_tokens and cached_req.info.num_tokens >= 0
					batch.num_tokens[req_id] = cached_req.info.num_tokens
					batch.total_tokens = next_total_tokens
					assert req_id not in batch.request_id_meta
					batch.request_id_meta[req_id] = cached_req.item.req.meta

				index += 1

			assert index - batch_index == batch.num_requests + num_nones
			assert batch.num_requests == len(batch.num_tokens) == len(batch.request_id_meta)
			assert batch.total_tokens == sum(batch.num_tokens)
			assert batch.num_tokens.keys() == batch.request_id_meta.keys()

			if (batch.full_batch or force) and batch.num_requests >= 1:

				def revert_state(max_batch_id: int, request_id_meta: dict[int, Optional[dict[str, Any]]], queue: list[Optional[CachedGPTRequest]]):
					self.Q[batch_index:index] = queue
					self.S.queue.request_id_meta = request_id_meta
					if self.S.batches and self.S.batches[-1] is batch:
						self.S.batches.pop()
					self.S.max_batch_id = max_batch_id

				with utils.AtomicExitStack() as stack:

					stack.callback(revert_state, max_batch_id=self.S.max_batch_id, request_id_meta=self.S.queue.request_id_meta.copy(), queue=self.Q[batch_index:index])

					self.S.max_batch_id += 1
					batch.id = self.S.max_batch_id
					batch.local_jsonl = os.path.join(self.working_dir, f"{self.name_prefix}_batch{batch.id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl")

					self.S.batches.append(batch)
					for req_id, req_meta in batch.request_id_meta.items():
						assert self.S.queue.request_id_meta[req_id] == req_meta
						del self.S.queue.request_id_meta[req_id]
					self.Q[batch_index:index] = (None,) * (index - batch_index)

					with utils.SafeOpenForWrite(path=batch.local_jsonl, stack=stack) as file:
						file.writelines(batch_reqs)
						file_size = utils.get_file_size(file)
					assert file_size == batch.local_jsonl_size
					log.info(f"{self.name_prefix}: Created batch {batch.id}, a {'full' if batch.full_batch else 'trailing'} local batch of size {batch.local_jsonl_size / 1000000:.3g}MB with {batch.num_requests} requests and {batch.total_tokens} tokens [{os.path.basename(batch.local_jsonl)}]")

					self.validate_state_queue(clean=False)
					self.queue.save(stack=stack)
					self.state.save(stack=stack)

					stack.pop_all()

				num_created += 1
				num_created_requests += batch.num_requests

		log.info(f"{self.name_prefix}: Created {num_created} batches from {num_created_requests} requests, leaving {self.PQ.queue_len} requests in the request queue for the future")
		return self.num_unpushed_batches()

	# Push as many batches as possible for remote processing and return whether the batch pipeline is currently congested (i.e. a certain number of batches are complete and pending but cannot be pushed yet due to thresholds)
	def push_batches(self) -> bool:
		# TODO: Log info about how many batches were pushed (one for each pushed batch with stats, e.g. batch ID and remote ID and file ID and such as well?)
		# TODO: Need to be totally safe that congested can only be True if not(RF)
		# TODO: Return whether batch pipeline is congested
		# TODO: Log info if return value is true (batch_congestion)
		# TODO: MUST: Update whatever state is necessary so that the output of num_unfinished_batches() is still correct without a call to update_remote_batch_state or whatever (don't really need to do anything actually?)
		# TODO: MUST: After this function the return value of num_unpushed_batches() must be correct
		# TODO: See what unpushed batches there are and push them
		# TODO: Also want to know exactly which push limit is reached when and why (explicit?)
		return CANNOT_PUSH_ANY_CURRENT_BATCH and self.num_unpushed_batches() >= self.max_unpushed_batches

	# Retrieve the number of unpushed local batches
	def num_unpushed_batches(self) -> int:
		return sum(1 for batch in self.S.batches if batch.remote is None)

	# Return how many remote batches there are
	def num_remote_batches(self) -> int:
		return sum(1 for batch in self.S.batches if batch.remote is not None)

	# Update the current state of all pushed remote batches
	def update_remote_batch_state(self, only_check_finished: bool) -> bool:
		# only_check_finished = If True, return as soon as a single pushed remote batch is seen to be in the finished state (whether this is new or not)
		# Returns whether there are any finished remote batches

		any_finished = False
		status_changed = False
		for batch in self.S.batches:
			if batch.remote is not None:

				if batch.remote.finished:
					any_finished = True
					if only_check_finished:
						break
				else:

					try:
						batch.remote.status = self.client.batches.retrieve(batch_id=batch.remote.batch_id)
						status_changed = True
						if batch.remote.status.status in ('failed', 'completed', 'expired', 'cancelled'):
							batch.remote.finished = True
							any_finished = True
							if only_check_finished:
								break
					except openai.OpenAIError as e:
						log.error(f"Failed to retrieve remote batch status with {utils.get_class_str(e)}: {e}")

		if status_changed:
			with utils.AtomicExitStack() as stack:
				self.validate_state_queue(clean=False)
				self.state.save(stack=stack)
				stack.pop_all()

		return any_finished

	# Return how many unfinished remote batches there are according to the latest local state of information (see update_remote_batch_state())
	def num_unfinished_batches(self) -> int:
		return sum(1 for batch in self.S.batches if batch.remote is not None and not batch.remote.finished)

	# Return how many finished remote batches there are according to the latest local state of information (see update_remote_batch_state())
	def num_finished_batches(self) -> int:
		return sum(1 for batch in self.S.batches if batch.remote is not None and batch.remote.finished)

	# Wait until there is at least one finished yet unprocessed remote batch or no more unfinished remote batches
	def wait_for_batches(self):

		initial_num_unfinished_batches = self.num_unfinished_batches()
		if self.num_finished_batches() > 0 or initial_num_unfinished_batches <= 0:
			return

		start_time = time.perf_counter()
		printed_str = None

		while True:
			self.update_remote_batch_state(only_check_finished=True)
			num_unfinished_batches = self.num_unfinished_batches()
			if self.num_finished_batches() > 0 or num_unfinished_batches <= 0:
				utils.print_in_place(f"{self.name_prefix}: Waited {utils.format_duration(time.perf_counter() - start_time)} until {initial_num_unfinished_batches - num_unfinished_batches} batch(es) finished\n")
				return
			time.sleep((start_time - time.perf_counter()) % self.remote_update_interval)
			print_str = f"{self.name_prefix}: Still waiting on {num_unfinished_batches} unfinished batches after {utils.format_duration(time.perf_counter() - start_time)}... "
			if print_str != printed_str:
				utils.print_in_place(print_str)
				printed_str = print_str

	# TODO: Does this need to be more like "retrieve finished batches"? Does it return a context manager? And iterable?
	def process_batches(self):  # TODO: Return value annotation (does this return num_unfinished_batches? Or rather (NOT?) how many it processed?)
		# TODO: MUST Use update_remote_batch_state()
		# TODO: Log error if a batch is finished by NOT completed => FAIL the associated samples (or retry them once auto-retrying is a thing) if remote_batch.status != 'completed': print(f"ERROR: Remote batch {remote_batch.id} has status '{remote_batch.status}' with errors: {remote_batch.errors}")
		# TODO: if remote_batch.output_file_id: remote_batch_content = tuple(json.loads(line) for line in client.files.content(file_id=remote_batch.output_file_id).text.splitlines() if line)
		# TODO: If you process batches, ensure that whatever state you change immediately reflects this in num_finished_batches (when you delete the remote batches they are no longer remote batches)
		# TODO: Remove processed batches from the server/batch state list/local batch file => That way self.num_finished_batches() is implicitly updated
		pass

	# TODO: Add transparent retry/attempts support (the function/code that processes received responses can indicate whether a retry is required, and then that happens if there are remaining attempts possible, otherwise permanent failure of the request)

	# TODO: Errors that one should not attempt to continue from (e.g. inconsistent state?) so that manual intervention can fix, errors like timeouts that could benefit from a simple retry (but need to back off?), errors like requests/samples that permanently fail and never go through - when to accept? How to retry after accepted? (allow a launch to clear the failed task manager state / gpt requester attempts state?)

	# TODO: BatchState:
	# TODO:   Need like num_requests, num_tokens, etc? Whatever is relevant for cutoffs/thresholds/decisions
	# TODO:   Always assert sizes when creating request_id_meta dicts (BatchState) to make sure no key overlap

	# TODO: Have a batch size threshold below which direct requests are used instead of Batch API (minimum batch size?)

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
	# TODO: Wandb (the entire 'metrics' part of the current state, plus how many batches are active etc) => ALL wandb parameters associated with each are updated EVERY time the task state, state, poolqueue are saved (three wandb.log statements only, essentially)
	# TODO: Any meaningful way to implement auto-retry individual failed requests? (up to a certain retry count)
	# TODO: Also need a feature to NOT keep retrying requests/samples indefinitely that are just failing (hard because they get reconstructed or might be batched where only 1 of 15 is the failing reason)

	# TODO: In GPT requester: WARN if input token estimations are far off individually, or definitely too low on average (+2%)
	# TODO: Have a dryrun mode that can be used to just see the cost of batches that WOULD be launched (NO actual state update - no cost, no file writes)

	# TODO: Allow threshold below which a forced batch/push automatically uses single API requests instead

	# TODO: wandb (conditional import to not require install if don't need it!)

	# TODO: Argparse suboptions (as namespace that can be passed to one of these classes) -> Don't need as hydra (have hydra suboptions?)? But good for others?
# EOF
