# OpenAI GPT requester

# Imports
import os
import logging
import contextlib
import dataclasses
from typing import Optional, Self, Any
from . import utils
from .logger import log

# Logging configuration
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

#
# GPT requester state file
#

# GPT requester batch state class
@dataclasses.dataclass
class GRBatchState:
	local_jsonl: str        # Path of the generated local input JSONL file for the batch
	local_jsonl_bytes: int  # Number of bytes of the local input JSONL file
	remote_jsonl_id: str    # Assigned string ID of the uploaded input JSONL file (OpenAI file storage)
	remote_batch: str       # Assigned string ID of the batch started based on the uploaded input JSONL file (OpenAI batches)

# GPT requester state class
@dataclasses.dataclass
class GRState:
	version: int = 1                                                       # Data format version number
	batches: list[GRBatchState] = dataclasses.field(default_factory=list)  # States of all batches that are currently pending or in progress

# GPT requester state file class
class GRStateFile:

	state: Optional[GRState]

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
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self):
		self.state = GRState()
		self.save()

	def load(self):
		with open(self.path, 'r') as file:
			file_size = utils.get_file_size(file)
			self.state = utils.dataclass_from_json(cls=GRState, json_data=file)
			log.info(f"Loaded GPT requester state file with {len(self.state.batches)} batches ({utils.format_size(file_size)}): {self.name}")

	def save(self):
		if self.state is not None:
			with utils.SafeOpenForWrite(self.path) as file:
				utils.json_from_dataclass(obj=self.state, file=file)
				file_size = utils.get_file_size(file)
			log.info(f"Saved GPT requester state file with {len(self.state.batches)} batches ({utils.format_size(file_size)}): {self.name}")

	def unload(self):
		self.save()
		self.state = None

#
# GPT requester requests file
#

# GPT request class
@dataclasses.dataclass(frozen=True)
class GPTRequest:
	payload: dict[str, Any]         # Request payload (JSON-compatible OpenAI request) => Is batched into JSONL files and uploaded to OpenAI for processing
	meta: Optional[dict[str, Any]]  # Optional custom metadata to associate with this request (JSON-compatible) => Is batched and stored in the state file while OpenAI is processing the payloads

# GPT requester requests class
@dataclasses.dataclass
class GRRequests:
	saved: list[Optional[GPTRequest]] = dataclasses.field(default_factory=list)  # List of unbatched pending requests that are currently backed by the requests file (None elements mean that the corresponding saved request is no longer required and should be removed from the requests file on the next save)
	unsaved: list[GPTRequest] = dataclasses.field(default_factory=list)          # List of unbatched pending requests that are not currently backed by the requests file (any new requests are added here, and then moved over to saved after a successful save)

# GPT requester requests file class
class GRRequestsFile:

	requests: Optional[GRRequests]

	def __init__(self, path: str):
		# path = Path to the JSONL requests file to load/save/manage (nominally *.jsonl extension)
		self.path = os.path.abspath(path)
		log.info(f"GPT requester requests file: {self.path}")
		self.name = os.path.basename(self.path)
		self._enter_stack = contextlib.ExitStack()
		self.requests = None

	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.callback(self.unload)
			try:
				self.load()
			except FileNotFoundError:
				self.create()
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self):
		self.requests = GRRequests()
		self.save()

	def load(self):
		with open(self.path, 'r') as file:
			file_size = utils.get_file_size(file)
			self.requests = GRRequests(saved=[utils.dataclass_from_json(cls=GPTRequest, json_data=line) for line in file.readlines()], unsaved=[])
			log.info(f"Loaded GPT requester requests file with {len(self.requests.saved)} requests ({utils.format_size(file_size)}): {self.name}")

	def save(self):
		if self.requests is not None:
			saved = [request for request in self.requests.saved if request is not None]
			saved.extend(self.requests.unsaved)
			with utils.SafeOpenForWrite(self.path) as file:
				for request in saved:
					utils.json_from_dataclass(obj=request, file=file)
					file.write('\n')
				file_size = utils.get_file_size(file)
			log.info(f"Saved GPT requester requests file with {len(saved)} requests ({utils.format_size(file_size)}): {self.name}")
			self.requests.saved[:] = saved
			self.requests.unsaved.clear()

	def unload(self):
		self.save()
		self.requests = None

#
# GPT requester
#

# GPT requester class
class GPTRequester:

	def __init__(
		self,
		working_dir: str,                              # Path to the GPT working directory to use (will be used for automatically managed lock, state, requests and batch files)
		autocreate_working_dir: bool = True,           # Whether to automatically create the GPT working directory if it does not exist (parent directory must already exist)
		name_prefix: str = 'gpt_requester',            # Name prefix to use for all files created in the GPT working directory
		lock_timeout: Optional[float] = None,          # Timeout (if any) to use when attempting to lock exclusive access to the files in the GPT working directory corresponding to the given name prefix (see utils.LockFile)
		lock_poll_interval: Optional[float] = None,    # Lock file polling interval (see utils.LockFile)
		lock_status_interval: Optional[float] = None,  # Lock file status update interval (see utils.LockFile)
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
		self.state = GRStateFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_state.json"))
		self.requests = GRRequestsFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_requests.jsonl"))

		self._enter_stack = contextlib.ExitStack()

	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.enter_context(self.lock)
			stack.enter_context(self.state)
			stack.enter_context(self.requests)
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)


	# TODO: If there is a standalone action with no synchronicity required with any other action, then being atomic is enough
	# TODO: Otherwise, everything that requires state synchronicity should be linked inside an exitstack that undoes EVERYTHING perfectly if anything fails (this could mean keeping a backup copy of a state file before it was successfully changed, in case a later synchronised action errors)
	# TODO: For convenience, such exit stacks should delay KeyboardInterrupt to avoid completely unnecessary reversions based on a badly timed user Ctrl+C
	# TODO: Summary: Either single atomic operation or multiple perfectly reversible ones in utils.DelayKeyboardInterruptStack

	# TODO: No exception (KeyboardInterrupt or Error) should leave the disk state in an inconsistent state (cannot really guarantee that when saving state/requests files sequentially?)
	# TODO: Load and sanity check the state file if it exists, otherwise create one, Load and sanity check the requests file (dual file contents vs added on top lists in memory) if it exists (error to exist if no state file existed), otherwise create an empty one

	@contextlib.contextmanager
	def add_request(self, req: GPTRequest) -> ContextManager[bool]:  # TODO: Auto-true flag whether to send off complete batches if possible, or literally just add it to the requests JSONL?
		# TODO: Need to yield whether any state actually changed yet?
		yield

	@contextlib.contextmanager
	def add_requests(self, reqs: Iterable[GPTRequest]) -> ContextManager[None]:  # TODO: Auto-true flag whether to send off complete batches if possible, or literally just add it to the requests JSONL?
		# TODO: Can only iterate Iterable ONCE!!!
		yield

	# TODO: Version
	# TODO: Metrics (tokens in/out (what happens about thought tokens), per batch, num requests, divided by successful/unsuccessful)
	# TODO: Wandb
	# TODO: Remove tqdm if it is not actually used (would be better for wandb logs)
	# TODO: Any meaningful way to implement auto-retry individual failed requests? (up to a certain retry count)
	# TODO: Also need a feature to NOT keep retrying requests/samples indefinitely that are just failing (hard because they get reconstructed or might be batched where only 1 of 15 is the failing reason)
# EOF
