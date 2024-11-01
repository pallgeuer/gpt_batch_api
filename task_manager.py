# Task manager (wraps a GPT requester to include the sample generation side of the pipeline as well)

# Imports
import os
import copy
import contextlib
import dataclasses
from typing import Any, Optional, Self
from .logger import log
from . import gpt_requester, utils

#
# Task state file
#

# Task state class
@dataclasses.dataclass
class TaskState:
	version: int = 1                                                             # Data format version number
	meta: dict[str, Any] = dataclasses.field(default_factory=dict)               # Task-level metadata
	committed_meta: dict[str, Any] = dataclasses.field(default_factory=dict)     # Task-level information about the samples committed so far (arbitrary key-value store)
	committed_samples: dict[str, Any] = dataclasses.field(default_factory=dict)  # Sample-level information about the samples committed so far (maps sample keys to corresponding JSON-compatible data)
	completed_meta: dict[str, Any] = dataclasses.field(default_factory=dict)     # Task-level information about the samples completed so far (arbitrary key-value store)
	completed_samples: dict[str, Any] = dataclasses.field(default_factory=dict)  # Sample-level information about the samples completed so far (maps sample keys to corresponding JSON-compatible data)

# Task state file class
class TaskStateFile:

	state: Optional[TaskState]

	def __init__(self, path: str, init_meta: Optional[dict[str, Any]]):
		# path = Path to the JSON task state file to load/save/manage (nominally *.json extension)
		# init_meta = Value to initialise the meta field with if the task state file is newly created (deep copy on create)
		self.path = os.path.abspath(path)
		log.info(f"Task state file: {self.path}")
		self.name = os.path.basename(self.path)
		self.init_meta = init_meta if init_meta is not None else {}
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
		self.state = TaskState(meta=copy.deepcopy(self.init_meta))
		self.save(stack=None)

	def load(self):
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.state = utils.dataclass_from_json(cls=TaskState, json_data=file)
		log.info(f"Loaded task state file with {len(self.state.committed_samples)} committed and {len(self.state.completed_samples)} completed samples ({utils.format_size(file_size)}): {self.name}")

	def save(self, stack: Optional[contextlib.ExitStack[Optional[bool]]]):
		with utils.SafeOpenForWrite(self.path, stack=stack) as file:
			utils.json_from_dataclass(obj=self.state, file=file)
			file_size = utils.get_file_size(file)
		log.info(f"Saved task state file with {len(self.state.committed_samples)} committed and {len(self.state.completed_samples)} completed samples ({utils.format_size(file_size)}): {self.name}")

	def unload(self):
		self.state = None

#
# Task manager class
#

# Task manager class
class TaskManager:

	# Construct a task manager to make use of the OpenAI Batch API to process samples
	def __init__(
		self,
		task_dir: str,                               # Path to the task working directory to use (will be used for automatically managed lock, state, requests, batch, task state, and output files)
		name_prefix: str,                            # Name prefix to use for all files created in the task working directory (e.g. 'my_task')
		*,                                           # Keyword arguments only beyond here
		init_meta: Optional[dict[str, Any]] = None,  # Value to initialise the task state meta field with if the task state file is newly created (deep copy on create)
		**gpt_requester_kwargs,                      # Keyword arguments to be passed on to the internal GPTRequester instance
	):

		self.GR = gpt_requester.GPTRequester(working_dir=task_dir, name_prefix=name_prefix, **gpt_requester_kwargs)
		self.task = TaskStateFile(path=os.path.join(self.GR.working_dir, f"{self.GR.name_prefix}_task.json"), init_meta=init_meta)
		# TODO: Output file MUST be _output*.EXT => Class to generically wrap what kind of output file? Or just a method override (would this require frequent duplication of boiler plate code for saving e.g. just a standard JSON)? / Generic TaskOutputFile (could be simple JSON, or potentially chunked JSONL, or totally different file ext?)

		self._enter_stack = contextlib.ExitStack()
		self.T: Optional[TaskState] = None

	# Run the task manager to completion
	def run(self):
		with self:
			raise NotImplementedError  # TODO: Is it possible to have this be a while True: step(); wait() where step() does everything it can up to the point where it would have to wait for a batch to complete?

	# Enter method for the required use of TaskManager as a context manager
	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.enter_context(self.GR)
			stack.callback(self.on_exit)
			stack.enter_context(self.task)
			self.T = self.task.state
			self.validate_state()
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	# Exit method for the required use of TaskManager as a context manager
	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	# Local actions to perform on exit
	def on_exit(self):
		self.T = None

	# Validate that there are no obvious issues with the current state
	def validate_state(self):
		if not self.T.committed_samples.keys() >= self.T.completed_samples.keys():
			raise ValueError(f"Unexpected completed yet not committed samples: {sorted(self.T.completed_samples.keys() - self.T.committed_samples.keys())}")
# EOF
