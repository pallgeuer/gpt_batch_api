# Task manager (wraps a GPT requester to include the sample generation side of the pipeline as well)

# Imports
import os
import copy
import json
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
	meta: dict[str, Any] = dataclasses.field(default_factory=dict)               # Task-level metadata (arbitrary JSON-compatible key-value store, e.g. can be used to store parameters that should/must not change throughout an entire task, even if the task is resumed with different configuration arguments)
	committed_meta: dict[str, Any] = dataclasses.field(default_factory=dict)     # Task-level information about the samples committed so far (arbitrary JSON-compatible key-value store)
	committed_samples: dict[str, Any] = dataclasses.field(default_factory=dict)  # Sample-level information about the samples committed so far (maps sample keys to corresponding JSON-compatible data)
	responded_meta: dict[str, Any] = dataclasses.field(default_factory=dict)     # Task-level information about the samples that have received a response so far (arbitrary JSON-compatible key-value store)
	failed_samples: dict[str, Any] = dataclasses.field(default_factory=dict)     # Sample-level information about the committed samples that have received a response so far and failed (maps sample keys to corresponding JSON-compatible data)
	succeeded_samples: dict[str, Any] = dataclasses.field(default_factory=dict)  # Sample-level information about the committed samples that have received a response so far and succeeded (maps sample keys to corresponding JSON-compatible data)

# Task state file class
class TaskStateFile:

	state: Optional[TaskState]

	def __init__(self, path: str, reinit_meta: bool, init_meta: Optional[dict[str, Any]]):
		# path = Path to the JSON task state file to load/save/manage (nominally *.json extension)
		# reinit_meta = Whether to force a reinitialisation of the meta field even if the task state file already exists
		# init_meta = Value to initialise the meta field with if the task state file is newly created (deep copy on create)
		self.path = os.path.abspath(path)
		log.info(f"Task state file: {self.path}")
		self.name = os.path.basename(self.path)
		self.reinit_meta = reinit_meta
		self.init_meta = init_meta if init_meta is not None else {}
		self._enter_stack = contextlib.ExitStack()
		self.state = None

	# noinspection PyUnusedLocal
	def clear_reinit_meta(self, exc_type, exc_val, exc_tb) -> bool:
		if exc_type is None:
			self.reinit_meta = False
		return False

	def __enter__(self) -> Self:
		with self._enter_stack as stack, utils.AtomicExitStack() as atomic_stack:
			stack.callback(self.unload)
			try:
				self.load()
				if self.reinit_meta:
					self.state.meta.clear()
					self.state.meta.update(copy.deepcopy(self.init_meta))
					self.save(stack=atomic_stack)
			except FileNotFoundError:
				self.create(stack=atomic_stack)
			atomic_stack.push(self.clear_reinit_meta)
			assert self.state is not None
			log.info(f"Task metadata:\n{'\n'.join(f'    {key} = {json.dumps(value, ensure_ascii=False, indent=None)}' for key, value in self.state.meta.items())}")
			self._enter_stack = stack.pop_all()
		assert self._enter_stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self, stack: contextlib.ExitStack[Optional[bool]]):
		self.state = TaskState(meta=copy.deepcopy(self.init_meta))
		self.save(stack=stack)

	def load(self):
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.state = utils.dataclass_from_json(cls=TaskState, json_data=file)
		log.info(f"Loaded task state file with {len(self.state.committed_samples)} committed, {len(self.state.failed_samples)} failed, and {len(self.state.succeeded_samples)} succeeded sample keys ({utils.format_size(file_size)}): {self.name}")

	def save(self, stack: contextlib.ExitStack[Optional[bool]]):
		with utils.SafeOpenForWrite(self.path, stack=stack) as file:
			utils.json_from_dataclass(obj=self.state, file=file)
			file_size = utils.get_file_size(file)
		log.info(f"Saved task state file with {len(self.state.committed_samples)} committed, {len(self.state.failed_samples)} failed, and {len(self.state.succeeded_samples)} succeeded sample keys ({utils.format_size(file_size)}): {self.name}")

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
		reinit_meta: bool = False,                   # Whether to force a reinitialisation of the task state meta field even if the task state file already exists
		init_meta: Optional[dict[str, Any]] = None,  # Value to initialise the task state meta field with if the task state file is newly created (deep copy on create)
		**gpt_requester_kwargs,                      # Keyword arguments to be passed on to the internal GPTRequester instance
	):

		self.GR = gpt_requester.GPTRequester(working_dir=task_dir, name_prefix=name_prefix, **gpt_requester_kwargs)
		self.task = TaskStateFile(path=os.path.join(self.GR.working_dir, f"{self.GR.name_prefix}_task.json"), reinit_meta=reinit_meta, init_meta=init_meta)
		# TODO: Output file MUST be _output*.EXT => Class to generically wrap what kind of output file? Or just a method override (would this require frequent duplication of boiler plate code for saving e.g. just a standard JSON)? / Generic TaskOutputFile (could be simple JSON, or potentially chunked JSONL, or totally different file ext?)

		self._enter_stack = contextlib.ExitStack()
		self.T: Optional[TaskState] = None
		self.step_num: Optional[int] = None

	# Run the task manager to completion
	def run(self):
		log.info(f"Running task manager {self.GR.name_prefix}...")
		with self:
			while self.step():
				pass  # TODO: Need to wait for at least one batch to finish and/or for there to be no batch pending?
		log.info(f"Finished running task manager {self.GR.name_prefix}")

	# Enter method for the required use of TaskManager as a context manager
	def __enter__(self) -> Self:
		with self._enter_stack as stack:
			stack.enter_context(self.GR)
			stack.callback(self.on_exit)
			self.step_num = 0
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
		self.T = self.step_num = None

	# Validate that there are no obvious issues with the current state
	def validate_state(self):
		# Note: This method can be overridden to sanity check task-specific task state conditions
		if not self.T.committed_samples.keys() >= self.T.failed_samples.keys():
			raise ValueError(f"Unexpected failed yet not committed samples: {sorted(self.T.failed_samples.keys() - self.T.committed_samples.keys())}")
		if not self.T.committed_samples.keys() >= self.T.succeeded_samples.keys():
			raise ValueError(f"Unexpected succeeded yet not committed samples: {sorted(self.T.succeeded_samples.keys() - self.T.committed_samples.keys())}")

	# Execute a single non-blocking step of the task manager (generates and processes as much as is currently possible without waiting)
	def step(self) -> bool:
		# Returns whether the task manager has work left to do, i.e. the request/batch pipeline is empty (no request is currently added, committed or pending a response) AND having called generate_requests() returned True (there are currently no requests to generate)

		self.step_num += 1
		log.info('-' * 80)
		log.info(f"{self.GR.name_prefix}: Step {self.step_num}...")

		work_ongoing = True
		while True:
			# TODO: Process any batches that are complete and update the task state accordingly => must return whether any batches are somehow in the pipeline...?
			log.info(f"{self.GR.name_prefix}: Generating requests...")
			generation_ongoing = self.generate_requests()  # TODO: If congested then this could always return True until no longer congested
			congested = self.commit_requests(force_batch=not generation_ongoing)
			# TODO: return True instead of wait (=> Generation is congested, because batches are congested), return False if all done

		log.info(f"{self.GR.name_prefix}: Finished step {self.step_num}")
		return work_ongoing  # TODO: Step shouldn't be able to finish with no pushed batches pending / with any unbatched requests (although some batches could be not pushed due to congestion)

	# Generate requests based on the current task state and add the requests to the GPT requester
	def generate_requests(self) -> bool:
		# Iterate through available samples and generate GPTRequest instances for work yet to do (i.e. requests that haven't previously already been generated and committed).
		# The task state, and in particular the committed_meta/committed_samples fields thereof (in combination with sample key strings) can be used to determine what work has already been generated, and what work is yet to do.
		# Return a boolean whether there is (potentially) more work that can be generated right now, e.g. because not all available samples were iterated through right now.
		# It must be the case that if generate_requests() is called repeatedly in succession it eventually returns False when there is nothing more that can be generated right now (e.g. at least until some currently in-progress requests potentially finish in the future).
		# Generated requests must contain enough metadata to allow unambiguous updating of the task state later in commit_cached_request(), i.e. at the very least the sample key(s) associated with the request, as well as any metadata required for later response processing and output file writing.
		# To allow samples to be committed and batched more incrementally (memory/disk consideration), it is permitted to manually call self.commit_requests() at any intermediate time in this method. Otherwise, it can be assumed that self.commit_requests() will be called automatically after this method returns.
		# If calling self.commit_requests() returns that the batch pipeline is currently congested (a certain number of batches are complete and pending but cannot be pushed yet due to thresholds), it is recommended to return early from this method and leave the generation of further requests to future calls of this method.
		raise NotImplementedError

	# Commit generated requests, and optionally batch and push them
	def commit_requests(self, batch: bool = True, force_batch: bool = False, push: bool = True) -> bool:
		# batch = Whether to batch requests after committing them
		# force_batch = Whether to force batching of all requests, i.e. whether to generate a trailing non-full batch with whatever requests are left
		# push = Whether to push batches (if possible) after creating them
		# Returns whether the batch pipeline is currently congested (always False if push=False)

		log.info(f"{self.GR.name_prefix}: Committing generated requests...")

		with self.GR.commit_requests() as (cached_reqs, stack):
			if cached_reqs:

				def revert_committed_state(meta: dict[str, Any], samples_all: bool, samples: dict[str, Any]):
					self.T.committed_meta.clear()
					self.T.committed_meta.update(meta)
					if samples_all:
						self.T.committed_samples.clear()
						self.T.committed_samples.update(samples)
					else:
						for sample_key, data in samples.items():
							if data is DELETE:
								self.T.committed_samples.pop(sample_key, None)
							else:
								self.T.committed_samples[sample_key] = data

				DELETE = object()
				if (sample_keys := self.cached_request_keys(cached_reqs)) is None:
					stack.callback(revert_committed_state, meta_=copy.deepcopy(self.T.committed_meta), samples_all=True, samples=copy.deepcopy(self.T.committed_samples))
				else:
					stack.callback(revert_committed_state, meta_=copy.deepcopy(self.T.committed_meta), samples_all=False, samples={sample_key: (copy.deepcopy(self.T.committed_samples[sample_key]) if sample_key in self.T.committed_samples else DELETE) for sample_key in sample_keys})

				for cached_req in cached_reqs:
					self.commit_cached_request(cached_req)
				self.validate_state()
				self.task.save(stack=stack)

		log.info(f"{self.GR.name_prefix}: Committed {len(cached_reqs)} generated requests")

		if batch:
			self.GR.batch_requests(force=force_batch)

		if push:
			congested = self.GR.push_batches()
		else:
			congested = False

		return congested

	# Extract from a list of CachedGPTRequest's a set of sample keys that is enough to cover all possible changes to the committed_samples task state when supplying these CachedGPTRequest's to commit_cached_request()
	# noinspection PyMethodMayBeStatic, PyUnusedLocal
	def cached_request_keys(self, cached_reqs: list[gpt_requester.CachedGPTRequest]) -> Optional[set[str]]:
		# cached_reqs = List of CachedGPTRequest's to extract all relevant sample key strings from
		# Return a set of sample key strings (the set or superset of sample key strings that will be modified by commit_cached_request()), or None (caller must assume all sample keys could be modified)
		return None

	# Update the committed_meta/committed_samples task state to reflect that a particular CachedGPTRequest has been committed
	def commit_cached_request(self, cached_req: gpt_requester.CachedGPTRequest):
		# cached_req = CachedGPTRequest that has been committed
		raise NotImplementedError

	# TODO: Calls some abstract methods to do the task-specific processing
	def process_batches(self):
		pass
# EOF
