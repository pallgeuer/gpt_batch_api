# OpenAI GPT batch API
#
# Note:
#  - The environment variable OPENAI_API_KEY is required for any OpenAI API requests to work
#  - The current remote files can be manually managed at: https://platform.openai.com/storage
#  - The current remote batches can be manually managed at: https://platform.openai.com/batches

# TODO: Organise and structure this file (e.g. utils.py)

# Imports
import os
import math
import time
import logging
import contextlib
import dataclasses
from typing import Optional, Iterable, Self, Any, Protocol, ContextManager
import tqdm
import filelock
import requests
import wandb
import openai
from . import utils
from .logger import log

# Logging configuration
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Protocol-based type annotation for configuration parameters (e.g. argparse.Namespace, omegaconf.DictConfig)
class Config(Protocol):

	def __getattr__(self, name: str) -> Any:
		...

	def __getitem__(self, key: str) -> Any:
		...

# TODO: Argparse suboptions (as namespace that can be passed to one of these classes) -> Don't need as hydra (have hydra suboptions?)? But good for others?
def foo(cfg: Config):
	log.info(f"Hey there! {cfg.foo}")
	requester = GPTRequester(working_dir='outputs/_gpt_batch_api/gpt_batch_api', name_prefix='prefix')
	with requester:
		log.info("ENTERED")
	log.info("EXITED")

# Lock file class
class LockFile:

	def __init__(self, lock_file: str, timeout: float = -1, poll_interval: float = 0.25, status_interval: float = 5.0):
		self.lock = filelock.FileLock(lock_file=lock_file)
		self.timeout = timeout
		self.poll_interval = poll_interval
		if self.poll_interval < 0.01:
			raise ValueError(f"Poll interval must be at least 0.01s: {self.poll_interval}")
		self.status_interval = status_interval
		if self.status_interval < 0.01:
			raise ValueError(f"Status interval must be at least 0.01s: {self.status_interval}")

	def __enter__(self) -> Self:
		start_time = time.perf_counter()
		print_time = start_time + self.status_interval
		if self.timeout >= 0:
			timeout_time = start_time + self.timeout
			timeout_str = f'/{utils.format_duration_unit(self.timeout)}'
		else:
			timeout_time = math.inf
			timeout_str = ''
		now = start_time
		while True:
			try:
				self.lock.acquire(timeout=max(min(timeout_time, print_time) - now, 0), poll_interval=self.poll_interval)
				utils.print_clear_line()
				return self
			except filelock.Timeout:
				now = time.perf_counter()
				if now >= timeout_time:
					utils.print_in_place(f"Failed to acquire lock in {utils.format_duration_unit(now - start_time)}: {self.lock.lock_file}\n")
					raise
				elif now >= print_time:
					utils.print_in_place(f"Still waiting on lock after {utils.format_duration_unit(now - start_time)}{timeout_str}: {self.lock.lock_file} ")
					print_time += self.status_interval

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		self.lock.release()
		return False

# GPT request class
@dataclasses.dataclass(frozen=True)
class GPTRequest:
	payload: dict[str, Any]         # Request payload (JSON-like)
	meta: Optional[dict[str, Any]]  # Optional custom metadata to associate with this request (JSON-like)

# GPT requester class
class GPTRequester:

	def __init__(self, working_dir: str, autocreate_working_dir: bool = True, name_prefix: str = 'gpt_requester', lock_timeout: float = -1):
		# TODO: DOC

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

		self.lock_timeout = lock_timeout
		self.lock_file = os.path.join(self.working_dir, f"{self.name_prefix}.lock")
		self.state_file = os.path.join(self.working_dir, f"{self.name_prefix}_state.json")
		self.requests_file = os.path.join(self.working_dir, f"{self.name_prefix}_requests.json")
		self.batch_file_prefix = os.path.join(self.working_dir, f"{self.name_prefix}_batch_")

		self._enter_stack = contextlib.ExitStack()
		self.lock = LockFile(lock_file=self.lock_file)
		self.state: Optional[dict[str, Any]] = None
		self.requests = None  # TODO: requests should be a dataclass pair of lists corresponding to those requests that are currently on disk in the file, and temporary ones that have been added in code at runtime but not saved yet

	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.enter_context(self.lock)
			with utils.DelayKeyboardInterrupt():
				stack.callback(self.unload_state_file)  # TODO: Just set some state members to None etc
				try:
					self.load_state_file()
				except FileNotFoundError:
					self.create_state_file()
				stack.callback(self.unload_requests_file)  # TODO: Just set some state members to None etc
				try:
					self.load_requests_file()
				except FileNotFoundError:
					self.create_requests_file()
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create_state_file(self):
		# TODO: Further implement load_state_file, unload_state_file, create_requests_file, load_requests_file, unload_requests_file
		self.state = dict(  # TODO: Dataclass-based
			version=1,
			batches=[],
		)



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
# EOF
