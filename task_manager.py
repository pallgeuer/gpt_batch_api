# Task manager (wraps a GPT requester to include sample generation and output file management)

# Imports
from __future__ import annotations
import os
import re
import copy
import json
import argparse
import functools
import contextlib
import collections
import dataclasses
import typing
from typing import Any, Optional, Self, Type, Callable, Generic, Iterable, Union
from .logger import log
from . import gpt_requester, utils

# Type variables
DataclassT = typing.TypeVar('DataclassT')

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
	failed_samples: dict[str, Any] = dataclasses.field(default_factory=dict)     # Sample-level information about the committed samples that have received a response so far and failed (maps sample keys to corresponding JSON-compatible data, keys MUST be a non-strict subset of committed_samples at all times)
	succeeded_samples: dict[str, Any] = dataclasses.field(default_factory=dict)  # Sample-level information about the committed samples that have received a response so far and succeeded (maps sample keys to corresponding JSON-compatible data, keys MUST be a non-strict subset of committed_samples at all times)

# Task state file class
class TaskStateFile:

	state: Optional[TaskState]

	def __init__(self, path: str, reinit_meta: bool, init_meta: Optional[dict[str, Any]], dryrun: bool):
		# path = Path to the JSON task state file to load/save/manage (nominally *.json extension)
		# reinit_meta = Whether to force a reinitialisation of the meta field even if the task state file already exists
		# init_meta = Value to initialise the meta field with if the task state file is newly created (deep copy on create)
		# dryrun = Whether to prevent any saving of state (dry run mode)
		self.path = os.path.abspath(path)
		self.name = os.path.basename(self.path)
		log.info(f"Task state file: {self.path}")
		self.reinit_meta = reinit_meta
		self.init_meta = init_meta if init_meta is not None else {}
		self.dryrun = dryrun
		self._enter_stack = contextlib.ExitStack()
		self.state = None

	# noinspection PyUnusedLocal
	def clear_reinit_meta(self, exc_type, exc_val, exc_tb) -> bool:
		if exc_type is None:
			self.reinit_meta = False
		return False

	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicRevertStack() as rstack:
				enter_stack.callback(self.unload)
				try:
					self.load(rstack=rstack)
					if self.reinit_meta:
						self.state.meta = copy.deepcopy(self.init_meta)
						self.save(rstack=rstack)
				except FileNotFoundError:
					self.create(rstack=rstack)
				rstack.push_always(self.clear_reinit_meta)
				assert self.state is not None
				log.info(f"Task metadata:{''.join(f'\n    {key} = {json.dumps(value, ensure_ascii=False, indent=None)}' for key, value in self.state.meta.items())}")
			self._enter_stack = enter_stack.pop_all()
		assert self._enter_stack is not enter_stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible creation of the task state file
		rstack.callback(setattr, self, 'state', self.state)
		self.state = TaskState(meta=copy.deepcopy(self.init_meta))
		self.save(rstack=rstack)

	def load(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible loading of the task state file
		rstack.callback(setattr, self, 'state', self.state)
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.state = utils.dataclass_from_json(cls=TaskState, json_data=file)
		log.info(f"Loaded task state file with {len(self.state.committed_samples)} committed, {len(self.state.failed_samples)} failed, and {len(self.state.succeeded_samples)} succeeded sample keys ({utils.format_size_iec(file_size)})")

	def save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible saving of the task state file
		if self.dryrun:
			log.warning(f"{gpt_requester.DRYRUN}Did not save task state file with {len(self.state.committed_samples)} committed, {len(self.state.failed_samples)} failed, and {len(self.state.succeeded_samples)} succeeded sample keys")
		else:
			with utils.SafeOpenForWrite(path=self.path, rstack=rstack) as file:
				utils.json_from_dataclass(obj=self.state, file=file)
				file_size = utils.get_file_size(file)
			log.info(f"Saved task state file with {len(self.state.committed_samples)} committed, {len(self.state.failed_samples)} failed, and {len(self.state.succeeded_samples)} succeeded sample keys ({utils.format_size_iec(file_size)})")

	def unload(self):
		self.state = None

#
# Task output file
#

# Task output file class
class TaskOutputFile:

	def __init__(self, path_base: str, dryrun: bool):
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output
		# dryrun = Whether to prevent any saving of output (dry run mode)
		self.path_base = os.path.abspath(path_base)
		self.dryrun = dryrun
		self.data = None

	def __enter__(self) -> Self:
		# Load/create the task output file and set/initialise self.data (must be mutable or None)
		raise NotImplementedError

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		# Unload the task output file (don't automatically just save the task output file here - it is the responsibility of the user of the class to explicitly call save() when required)
		raise NotImplementedError

	def validate(self):
		# Assuming the class is entered, perform any possible validations on the data and raise a ValueError for any failures
		pass

	def create(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible creation of the task output file
		# Initialise the task output file on disk and in memory
		raise NotImplementedError

	def reset(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible resetting of the task output file
		# Reset the task output file to the state it is in right after creation (don't necessarily assume that any self.data is currently already created/loaded)
		raise NotImplementedError

	def load(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible loading of the task output file
		# Load the task output file to memory
		raise NotImplementedError

	def save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible saving of the task output file
		# Save the current memory state of the task output file to disk
		# It is permitted to make changes to self.data (e.g. sorting keys of a dictionary or so) just prior to saving, as long as it is reversible (rstack)
		raise NotImplementedError

	def unload(self):
		# Unload the task output file
		self.data = None

# Dataclass output file class
class DataclassOutputFile(TaskOutputFile, Generic[DataclassT]):

	data: Optional[DataclassT]  # TaskOutputFile: Data backed by the output file (while the class is in the entered state)

	@classmethod
	def read(cls, path_base: str, data_cls: Optional[Type[DataclassT]] = None) -> Self:
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output
		# data_cls = Dataclass type to use (must be instantiatable without arguments, None = Assume cls.Dataclass exists and use it)
		# Returns a new instance of the class in read-only mode (raises an exception if load() fails on enter, or a save() is attempted)
		return cls(path_base=path_base, dryrun=True, data_cls=data_cls, read_only=True)

	@classmethod
	def output_factory(cls, data_cls: Optional[Type[DataclassT]] = None) -> Callable[[str, bool], DataclassOutputFile[DataclassT]]:
		# data_cls = Dataclass type to use (must be instantiatable without arguments, None = Assume cls.Dataclass exists and use it)
		# Returns a (not read-only mode) factory function suitable for passing as the output_factory argument of the TaskManager class
		return functools.partial(cls, data_cls=data_cls, read_only=False)

	def __init__(self, path_base: str, dryrun: bool, *, data_cls: Optional[Type[DataclassT]], read_only: bool):
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output (extension of .json will automatically be added)
		# dryrun = Whether to prevent any saving of output and just log what would have happened instead (dry run mode)
		# data_cls = Dataclass type to use (must be instantiatable without arguments, None = Assume cls.Dataclass exists and use it)
		# read_only = Whether to use read-only mode (raises an exception if load() fails on enter, or a save() is attempted)
		super().__init__(path_base=path_base, dryrun=dryrun)
		try:
			self.data_cls = data_cls if data_cls is not None else getattr(type(self), 'Dataclass')
		except AttributeError:
			raise ValueError("If no explicit dataclass type is provided via data_cls=X, then this class must be a subclass with a defined 'Dataclass' class attribute")
		self.read_only = read_only
		self.path = f'{self.path_base}.json'
		self.name = os.path.basename(self.path)
		log.info(f"Task output file: {self.path}")
		self._enter_stack = contextlib.ExitStack()

	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicRevertStack() as rstack:
				enter_stack.callback(self.unload)
				try:
					self.load(rstack=rstack)
				except FileNotFoundError:
					if self.read_only:
						raise
					else:
						self.create(rstack=rstack)
				assert self.data is not None
			self._enter_stack = enter_stack.pop_all()
		assert self._enter_stack is not enter_stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def validate(self):
		if not isinstance(self.data, self.data_cls):
			raise ValueError(f"Data is of unexpected type: {utils.get_class_str(type(self.data))} vs {utils.get_class_str(self.data_cls)}")

	def create(self, rstack: utils.RevertStack):
		rstack.callback(setattr, self, 'data', self.data)
		self.data = self.data_cls()
		self.save(rstack=rstack)

	def reset(self, rstack: utils.RevertStack):
		self.create(rstack=rstack)

	def load(self, rstack: utils.RevertStack):
		rstack.callback(setattr, self, 'data', self.data)
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.data = utils.dataclass_from_json(cls=self.data_cls, json_data=file)
		log.info(f"Loaded task output file with {self.status_str()} ({utils.format_size_iec(file_size)})")

	def pre_save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use to make any changes to the data reversible
		# This method can be overridden to perform changes to self.data immediately prior to each save (e.g. sorting keys of a dictionary or so)
		pass

	def save(self, rstack: utils.RevertStack):
		if self.read_only:
			raise RuntimeError("Cannot save dataclass output file in read-only mode")
		self.pre_save(rstack=rstack)
		if self.dryrun:
			log.warning(f"{gpt_requester.DRYRUN}Did not save task output file with {self.status_str()}")
		else:
			with utils.SafeOpenForWrite(path=self.path, rstack=rstack) as file:
				utils.json_from_dataclass(obj=self.data, file=file)
				file_size = utils.get_file_size(file)
			log.info(f"Saved task output file with {self.status_str()} ({utils.format_size_iec(file_size)})")

	def status_str(self) -> str:
		# Returns a string summarizing the data status for logging purposes (intended to be overridden by subclasses, "... task output file with STATUS_STR")
		return f"{len(dataclasses.fields(self.data))} fields"

# Dataclass list output file class
class DataclassListOutputFile(TaskOutputFile, Generic[DataclassT]):

	class NoFilesError(Exception):
		pass

	@dataclasses.dataclass
	class Data:
		paths: list[str]           # Ordered list of the paths of all current output files (there is always at least one path)
		last_entries: int          # Current number of entries in the last output file (on disk)
		last_size: int             # Current file size in bytes of the last output file (on disk)
		entries: list[DataclassT]  # Data read from or to be written to the output files

	data: Optional[Data]  # TaskOutputFile: Current data state (while the class is in the entered state)

	@classmethod
	def read(cls, path_base: str, data_cls: Optional[Type[DataclassT]] = None) -> Self:
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output
		# data_cls = Dataclass type to use for each list entry (None = Assume cls.Dataclass exists and use it)
		# Returns a new instance of the class in read-only mode (loads all entries from all output files, raises an exception if load() fails on enter or a save() is attempted)
		return cls(path_base=path_base, dryrun=True, data_cls=data_cls, read_only=True, max_entries=0, max_size=0)

	@classmethod
	def output_factory(cls, data_cls: Optional[Type[DataclassT]] = None, max_entries: int = 0, max_size: int = 0) -> Callable[[str, bool], DataclassListOutputFile[DataclassT]]:
		# data_cls = Dataclass type to use for each list entry (None = Assume cls.Dataclass exists and use it)
		# max_entries = Maximum number of entries to save per output file chunk (<=0 = No maximum)
		# max_size = Maximum file size in bytes per output file chunk (<=0 = No maximum)
		# Returns a (not read-only mode) factory function suitable for passing as the output_factory argument of the TaskManager class
		return functools.partial(cls, data_cls=data_cls, read_only=False, max_entries=max_entries, max_size=max_size)

	def __init__(self, path_base: str, dryrun: bool, *, data_cls: Optional[Type[DataclassT]], read_only: bool, max_entries: int = 0, max_size: int = 0):
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output (suffix of .jsonl or _partXofY.jsonl will automatically be added as appropriate)
		# dryrun = Whether to prevent any saving of output and just log what would have happened instead (dry run mode)
		# data_cls = Dataclass type to use for each list entry (None = Assume cls.Dataclass exists and use it)
		# read_only = Whether to use read-only mode (loads all entries from all output files, raises an exception if load() fails on enter or a save() is attempted)
		# max_entries = Maximum number of entries to save per output file chunk (<=0 = No maximum, not relevant in read-only mode)
		# max_size = Maximum file size in bytes per output file chunk (<=0 = No maximum, not relevant in read-only mode)
		super().__init__(path_base=path_base, dryrun=dryrun)
		try:
			self.data_cls = data_cls if data_cls is not None else getattr(type(self), 'Dataclass')
		except AttributeError:
			raise ValueError("If no explicit dataclass type is provided via data_cls=X, then this class must be a subclass with a defined 'Dataclass' class attribute")
		self.read_only = read_only
		self.max_entries = max_entries
		self.max_size = max_size
		log.info(f"Task output file(s): {self.path_base}*.jsonl")
		self.path_dirname = os.path.dirname(self.path_base)
		self.path_basename = os.path.basename(self.path_base)
		if not self.path_dirname or not self.path_basename:
			raise ValueError("Cannot have empty path dirname or basename")
		self._enter_stack = contextlib.ExitStack()

	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicRevertStack() as rstack:
				enter_stack.callback(self.unload)
				try:
					self.load(rstack=rstack)
				except type(self).NoFilesError:
					if self.read_only:
						raise
					else:
						self.create(rstack=rstack)
				assert self.data is not None
			self._enter_stack = enter_stack.pop_all()
		assert self._enter_stack is not enter_stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def validate(self):
		if not self.data.paths:
			raise ValueError("No output file paths")
		elif self.data.last_entries < 0 or self.data.last_size < 0:
			raise ValueError(f"Unexpected last output file state: {self.data.last_entries} entries, {self.data.last_size} file size")
		elif not all(isinstance(entry, self.data_cls) for entry in self.data.entries):
			raise ValueError(f"Not all entries are of the expected type: {utils.get_class_str(self.data_cls)}")

	def create(self, rstack: utils.RevertStack):
		if self.read_only:
			raise RuntimeError("Cannot create dataclass list output file in read-only mode")
		rstack.callback(setattr, self, 'data', self.data)
		self.data = type(self).Data(paths=[path := f'{self.path_base}.jsonl'], last_entries=0, last_size=0, entries=[])
		if self.dryrun:
			log.warning(f"{gpt_requester.DRYRUN}Did not create empty initial task output file")
		else:
			with utils.SafeOpenForWrite(path=path, rstack=rstack) as file:
				file_size = utils.get_file_size(file)
			assert file_size == self.data.last_size == 0
			log.info(f"Created empty initial task output file: {os.path.basename(path)}")

	def reset(self, rstack: utils.RevertStack):
		if self.data is not None:
			for path in self.data.paths:
				utils.safe_unlink(path=path, rstack=rstack)
		self.create(rstack=rstack)

	def load(self, rstack: utils.RevertStack):

		part_files = collections.defaultdict(set)
		for entry in os.listdir(self.path_dirname):
			entry_path = os.path.join(self.path_dirname, entry)
			if entry_path.startswith(self.path_base):
				entry_suffix = entry_path[len(self.path_base):]
				if entry_suffix.endswith('.jsonl') and os.path.isfile(entry_path):
					if entry_suffix == '.jsonl':
						part_num = part_total = 1
					elif match := re.fullmatch(r'_part([0-9]+)of([0-9]+)\.jsonl', entry_suffix):
						part_num = int(match.group(1))
						part_total = int(match.group(2))
					else:
						continue
					part_files[part_total].add((part_num, entry_path))

		if not part_files:
			raise type(self).NoFilesError("No output files exist that can be loaded")
		elif len(part_files) != 1:
			raise RuntimeError("Output file parts of differing part totals exist")
		part_total, parts_set = part_files.popitem()
		if part_total < 1:
			raise RuntimeError(f"Output file parts exist for an invalid part total of {part_total}")
		parts_list = sorted(parts_set)
		if len(parts_list) != part_total or tuple(part_num for part_num, entry_path in parts_list) != tuple(range(1, part_total + 1)):
			raise RuntimeError(f"Inconsistent or incomplete output file parts exist for a part total of {part_total}")

		rstack.callback(setattr, self, 'data', self.data)
		self.data = type(self).Data(paths=[entry_path for part_num, entry_path in parts_list], last_entries=0, last_size=0, entries=[])
		assert len(self.data.paths) == part_total >= 1

		if self.read_only:
			for path in self.data.paths:
				with open(path, 'r', encoding='utf-8') as file:
					self.data.entries.extend(utils.dataclass_from_json(cls=self.data_cls, json_data=line) for line in file)

		with open(self.data.paths[-1], 'r', encoding='utf-8') as file:
			self.data.last_entries = sum(1 for line in file)  # noqa
			self.data.last_size = utils.get_file_size(file)

		log.info(f"Loaded the task output file(s) in {len(self.data.paths)} parts")

	def pre_save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use to make any changes to the data entries reversible
		# This method can be overridden to perform changes to self.data.entries immediately prior to them being saved and cleared (e.g. sorting keys of a dictionary or so)
		pass

	def save(self, rstack: utils.RevertStack):

		if self.read_only:
			raise RuntimeError("Cannot save dataclass list output file in read-only mode")

		self.pre_save(rstack=rstack)

		Data = type(self).Data
		def revert_data(data: Data):  # noqa
			self.data.paths[:] = data.paths
			self.data.last_entries = data.last_entries
			self.data.last_size = data.last_size
			self.data.entries[:] = data.entries
		rstack.callback(revert_data, data=copy.deepcopy(self.data))

		if self.dryrun:
			log.warning(f"{gpt_requester.DRYRUN}Did not append {len(self.data.entries)} entries to the task output file(s)")
		else:

			added_entry_size = 0
			last_lines = []

			def save_last_lines():
				nonlocal added_entry_size
				with utils.SafeOpenForWrite(path=self.data.paths[-1], mode='ab', rstack=rstack) as file:
					file.writelines(last_lines)
					added_entry_size += sum(len(line) for line in last_lines)
				last_lines.clear()

			for entry in self.data.entries:
				entry_line = (utils.json_from_dataclass(obj=entry, indent=None) + '\n').encode('utf-8')
				entry_size = len(entry_line)
				if self.data.last_entries + 1 > self.max_entries > 0 or self.data.last_size + entry_size > self.max_size > 0:
					if last_lines:
						save_last_lines()
					new_num_paths = len(self.data.paths) + 1
					self.data.paths.append(f'{self.path_base}_part{new_num_paths:03d}of{new_num_paths:03d}.jsonl')  # Note: If this line executes then last_lines will be non-empty further below, and thus the file will be saved later on and will exist for sure with at least one entry
					self.data.last_entries = 0
					self.data.last_size = 0
					if entry_size > self.max_size > 0:
						raise ValueError(f"Entry of size {entry_size} cannot be appended to the currently EMPTY task output file due to over-restrictive max size of {self.max_size}")
				last_lines.append(entry_line)
				self.data.last_entries += 1
				self.data.last_size += entry_size

			if last_lines:
				save_last_lines()

			if len(self.data.paths) > 1:
				for p, path in enumerate(self.data.paths):
					correct_path = f'{self.path_base}_part{p + 1:03d}of{len(self.data.paths):03d}.jsonl'
					if path != correct_path:
						os.replace(src=path, dst=correct_path)
						rstack.callback(os.replace, src=correct_path, dst=path)
						self.data.paths[p] = correct_path

			log.info(f"Appended {len(self.data.entries)} entries to the now {len(self.data.paths)} task output file(s) (added {utils.format_size_iec(added_entry_size)})")

		self.data.entries.clear()

	def entries(self) -> Iterable[DataclassT]:
		for path in self.data.paths:
			with open(path, 'r', encoding='utf-8') as file:
				for line in file:
					yield utils.dataclass_from_json(cls=self.data_cls, json_data=line)

	def all_entries(self) -> list[DataclassT]:
		return list(self.entries())

#
# Task manager class
#

# Task manager class
@utils.init_kwargs
class TaskManager:

	# This abstract base class manages a gpt_requester.GPTRequester instance to complete a certain custom task. Task progress is stored in a task state file as well as a task output file.
	# This is in addition to the files stored by gpt_requester.GPTRequester, and the task files are also only updated when the gpt_requester.GPTRequester has its lock file locked.
	#
	# In order to use this class, subclass it for a particular task and implement/override:
	#   - __init__(cfg: utils.Config) => Customize a call to super().__init__(..., **gpt_requester_kwargs) based on an attribute-based cfg (coming from either Python argparse or Hydra, see below) where gpt_requester_kwargs comes from gpt_requester.GPTRequester.get_kwargs(cfg)
	#   - wipe_unfinished()           => Wipe any unfinished (and optionally also failed) requests/samples from the in-memory task state
	#   - validate_state()            => Validate that there are no obvious issues with the current state (remember to call super())
	#   - generate_requests()         => Implement request generation based on current task state
	#   - commit_cached_request()     => Update the committed_meta/committed_samples task state to reflect that a particular request has been committed
	#   - cached_request_keys()       => Extract from a list of requests a set of sample keys that is enough to cover all possible changes to the committed_samples task state when later supplying these requests to commit_cached_request()
	#   - process_batch_result()      => Process a batch result and accordingly update the task state and output files (must be a perfectly reversible operation)
	#
	# In order to conveniently provide relevant command line arguments, use either:
	#   - gpt_requester.GPTRequester.configure_argparse() => Python argparse
	#   - config/gpt_batch_api.yaml                       => Hydra configuration parameters YAML
	#
	# Each task needs to define a task state format (in particular a method for constructing string sample keys, see TaskState) and a request metadata format (see gpt_requester.GPTRequest) that need to satisfy the following properties:
	#  - Given committed_samples (dict with keys given by the sample keys committed so far) and possibly committed_meta/responded_meta/failed_samples/succeeded_samples as well, it must be possible to unambiguously determine which requests need to be generated and committed for the remaining task (at least possibly until further responses come back triggering further requests, see generate_requests())
	#  - Given only a single generated request and no further context, it must be possible to correctly update the committed_meta/committed_samples task state to reflect that the request has now been committed (generally implies that each request metadata needs to include the corresponding sample key the request originates from, see commit_cached_request())
	#  - OPTIONAL: Given only a list of generated requests and no further context, it should be possible to establish a set of sample keys containing all keys of committed_samples that could possibly be added/modified by committing the requests (generally implies that each request metadata needs to include the corresponding sample key the request originates from, see cached_request_keys())
	#  - Given a gpt_requester.BatchResult (contains request and response) and no further context other than committed_meta/committed_samples, it must be possible to update responded_meta/failed_samples/succeeded_samples in a way compatible with how generate_requests() works (see process_batch_result())
	#  - Given just the task state, it should be possible to reduce the record of committed samples to just those for which a response was received so far
	#  - Given just the task state, it should be possible to reduce the record of committed/responded samples to just those for which a succeeded response was received so far (failed samples are wiped)
	#  - Samples/requests that end up as failed should not make any contribution to the output file (e.g. can be wiped or skipped without affecting the task output)
	#
	# The init_meta argument to __init__ specifies parameter values that should always remain fixed throughout a task, even across multiple runs (this behaviour can be manually overridden using reinit_meta).

	# Construct a task manager to make use of the OpenAI Batch API to process samples
	def __init__(
		self,
		task_dir: str,                                          # Path of the task working directory to use (will be used for automatically managed lock, state, requests, batch, task state, and output files)
		name_prefix: str,                                       # Name prefix to use for all files created in the task working directory (e.g. 'my_task')
		output_factory: Callable[[str, bool], TaskOutputFile],  # Factory callable to create the required task output file instance (str argument is the required output file path base, e.g. /path/to/NAME_PREFIX_output, bool argument is whether dry run mode is active)
		init_meta: Optional[dict[str, Any]],                    # Value to initialise the task state meta field with if the task state file is newly created (deep copy on create)
		*,                                                      # Keyword arguments only beyond here

		run: bool = True,                                       # Whether to execute steps when the task manager is run, or just show the status and return (e.g. run=False is useful in combination with wipe_*)
		wipe_failed: bool = False,                              # CAUTION: Wipe and forget all failed samples from the task state (implies wipe_requests, does not wipe task output, consider running the task with only_process=True prior to wiping)
		reinit_meta: bool = False,                              # CAUTION: Whether to force a reinitialisation of the task state meta field even if the task state file already exists (normally the task state meta field is only initialised once at the beginning of a task and remains fixed after that across all future runs)

		**gpt_requester_kwargs,                                 # Keyword arguments to be passed on to the internal GPTRequester instance
	):

		self.run_flag = run
		self.wipe_failed = wipe_failed
		if self.wipe_failed:
			gpt_requester_kwargs['wipe_requests'] = True

		self.GR = gpt_requester.GPTRequester(working_dir=task_dir, name_prefix=name_prefix, **gpt_requester_kwargs)
		self.task = TaskStateFile(path=os.path.join(self.GR.working_dir, f"{self.GR.name_prefix}_task.json"), reinit_meta=reinit_meta, init_meta=init_meta, dryrun=self.GR.dryrun)
		self.output = output_factory(os.path.join(self.GR.working_dir, f"{self.GR.name_prefix}_output"), self.GR.dryrun)  # Arguments: path_base (str), dryrun (bool)

		self._enter_stack = contextlib.ExitStack()
		self.step_num: Optional[int] = None
		self.generating = False
		self.T: Optional[TaskState] = None
		self.D: Optional[Any] = None

	# Configure an argparse parser to incorporate an argument group for the keyword arguments that can be passed to the init of this class
	@staticmethod
	def configure_argparse(
		parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup],  # noqa / Argument parser or group
		*,                                                                # Keyword arguments only beyond here
		title: Optional[str] = 'Task manager',                            # If parser is not already an argument group, the title to use for the created argument group
		description: Optional[str] = None,                                # If parser is not already an argument group, the description to use for the created argument group
		group_kwargs: Optional[dict[str, Any]] = None,                    # If parser is not already an argument group, the extra keyword arguments to use for the created argument group
		**defaults,                                                       # noqa / Keyword arguments that can be used to override individual default argument values
	) -> argparse._ArgumentGroup:                                         # noqa / Returns the passed or newly created argument group

		group = parser.add_argument_group(title=title, description=description, **(group_kwargs if group_kwargs is not None else {})) if isinstance(parser, argparse.ArgumentParser) else parser
		add_argument = functools.partial(utils.add_init_argparse, cls=TaskManager, parser=group, defaults=defaults)

		add_argument(name='run', help="Whether to execute steps when the task manager is run, or just show the status and return (e.g. --no_run is useful in combination with --wipe_*)")
		add_argument(name='wipe_failed', help="CAUTION: Wipe and forget all failed samples from the task state (implies wipe_requests, does not wipe task output, consider running the task with --only_process prior to wiping)")
		add_argument(name='reinit_meta', help="CAUTION: Whether to force a reinitialisation of the task state meta field even if the task state file already exists (normally the task state meta field is only initialised once at the beginning of a task and remains fixed after that across all future runs)")

		return group

	# Run the task manager to completion
	def run(self):

		log.info('-' * 80)
		log.info("Running task manager...")

		with self:

			self.log_status()
			self.GR.log_status()

			if self.run_flag:
				while self.step():  # Returns True exactly if condition E*not[D] = PBMF(GQ + C)(not[L] + not[R]), i.e. only if condition F is satisfied => There are no pushed remote batches that are finished yet unprocessed
					if self.GR.dryrun:
						log.warning(f"{gpt_requester.DRYRUN}Stopping incomplete task manager as it is a dry run")
						break
					elif self.GR.num_unfinished_batches() <= 0:  # If condition RF...
						log.warning("Stopping incomplete task manager as a step did not result in unfinished pushed remote batches")
						break
					else:  # Else if condition not[R]F...
						self.GR.wait_for_batches()  # Waits (if not a dry run) until condition R + not[F] (nullifies condition F) => When this returns there must be at least one finished yet unprocessed remote batch, or no unfinished and/or unprocessed remote batches at all
			else:
				log.warning("Not running the task manager due to 'run' flag being False")

			log.info('-' * 80)
			self.log_status()
			self.GR.log_status()

		log.info("Finished running task manager")

	# Enter method for the required use of TaskManager as a context manager
	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			wipe_requests = self.GR.wipe_requests
			wipe_task = self.GR.wipe_task
			enter_stack.enter_context(self.GR)
			enter_stack.callback(self.on_exit)
			self.step_num = 0
			self.generating = False
			enter_stack.enter_context(self.task)
			self.T = self.task.state
			self.validate_state(clean=False)
			enter_stack.enter_context(self.output)
			self.D = self.output.data
			self.output.validate()
			self.wipe(wipe_requests=wipe_requests, wipe_task=wipe_task, wipe_failed=self.wipe_failed)  # Only does something if one of the arguments is True
			self._enter_stack = enter_stack.pop_all()
		assert self._enter_stack is not enter_stack
		return self

	# Exit method for the required use of TaskManager as a context manager
	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	# Local actions to perform on exit
	def on_exit(self):
		self.step_num = self.T = self.D = None
		self.generating = False

	# Perform a wipe of samples and/or the entire task
	def wipe(self, wipe_requests: bool, wipe_task: bool, wipe_failed: bool):
		# wipe_requests = Whether to wipe all ongoing requests and batches (primarily done by GPT requester already)
		# wipe_task = Whether to wipe the complete task state
		# wipe_failed = Whether to wipe the results of all failed samples (requires wipe_requests)
		# Wiping is NOT a revertible operation, and an indeterminate/inconsistent state in memory/on disk may result if an exception occurs during wiping

		if wipe_task:
			self.log_status()
			with utils.DelayKeyboardInterrupt():
				log.warning("Wiping complete task output and state...")
				with utils.RevertStack() as rstack:
					self.step_num = 0
					self.task.create(rstack=rstack)
					rstack.push_always(self.task.clear_reinit_meta)
					log.info(f"Task metadata:{''.join(f'\n    {key} = {json.dumps(value, ensure_ascii=False, indent=None)}' for key, value in self.task.state.meta.items())}")
					self.T = self.task.state
					self.validate_state(clean=True)
					self.output.reset(rstack=rstack)
					self.D = self.output.data
					self.output.validate()
				log.warning("Wiped complete task output and state")

		elif wipe_requests or wipe_failed:
			if not wipe_requests:
				raise ValueError("Wipe failed samples requires all ongoing requests also be wiped")  # As not wipe_requests, it is assumed that the GPTRequester has NOT wiped requests, and thus it is okay to raise an exception (does not result in indeterminate state on disk)
			self.log_status()
			with utils.DelayKeyboardInterrupt():
				log.warning(f"Wiping unfinished{' and failed' if wipe_failed else ''} requests/samples...")
				with utils.RevertStack() as rstack:
					self.wipe_unfinished(wipe_failed=wipe_failed)
					self.validate_state(clean=True)
					self.task.save(rstack=rstack)
				log.warning(f"Wiped unfinished{' and failed' if wipe_failed else ''} requests/samples")

	# Wipe any unfinished (and optionally also failed) requests/samples from the in-memory task state (self.T)
	def wipe_unfinished(self, wipe_failed: bool):
		# wipe_failed = Whether in addition to the committed-yet-unfinished samples to also wipe the failed samples (i.e. give them another chance)
		# The implementation should update the task state (self.T), specifically the committed_samples/committed_meta fields, as well as the failed_samples/responded_meta fields if wipe_failed
		# The task output should not need to be updated at all as unfinished and failed requests/samples should never affect the task output at all (requirement of the task design)
		# If not wipe_failed, this method should reduce the record of committed samples in self.T to just those committed samples that are responsible for the responses received so far (succeeded or failed)
		# If wipe_failed, this method should reduce the record of committed/responded samples in self.T to just those committed samples that are responsible for the succeeded samples (failed samples are also cleared)
		# The task state after this method finishes must pass self.validate_state(clean=True)
		raise NotImplementedError

	# Validate that there are no obvious issues with the current state
	def validate_state(self, *, clean: bool):
		# clean = Whether the current state should be without any unfinished requests/samples (all committed samples should have a response already)
		# Note: This method can be overridden to sanity check task-specific task state conditions (remember to call this implementation via super() though)
		if not self.T.committed_samples.keys() >= self.T.failed_samples.keys():
			raise ValueError(f"Unexpected failed yet not committed samples: {sorted(self.T.failed_samples.keys() - self.T.committed_samples.keys())}")
		if not self.T.committed_samples.keys() >= self.T.succeeded_samples.keys():
			raise ValueError(f"Unexpected succeeded yet not committed samples: {sorted(self.T.succeeded_samples.keys() - self.T.committed_samples.keys())}")
		if clean:
			responded_sample_keys = self.T.succeeded_samples.keys() | self.T.failed_samples.keys()
			if self.T.committed_samples.keys() != responded_sample_keys:
				raise ValueError(f"Unexpected committed relative to responded samples, with disagreement in the keys: {sorted(self.T.committed_samples.keys() ^ responded_sample_keys)}")

	# Log the current task manager status
	def log_status(self):
		num_ongoing = len(self.T.committed_samples.keys() - self.T.failed_samples.keys() - self.T.succeeded_samples.keys())
		num_done = len(self.T.committed_samples) - num_ongoing
		log.info(f"There are {len(self.T.committed_samples)} committed ({num_ongoing} ongoing and {num_done} done), {len(self.T.failed_samples)} failed, {len(self.T.succeeded_samples)} succeeded SAMPLES")
		assert num_done == len(self.T.failed_samples) + len(self.T.succeeded_samples)

	# Execute a single non-blocking step of the task manager (generates and processes as much as is currently possible without waiting)
	def step(self) -> bool:
		# Returns whether the task manager has work left to do
		#
		# Assumption: The available samples, on the basis of which generate_requests() generates requests, are fixed for the entire duration that a task manager is entered, e.g. for an entire call to run()
		# Assumption: When finished remote batches are processed, they do not have any direct influence on the push limits anymore, and thus on whether the batch pipeline is congested or not
		#
		# Conditions are boolean variables that are true if the condition is definitely true, and false if something occurs (nullification) that could POSSIBLY make the condition untrue, and rechecking is needed to have it set to true again (all conditions start as untrue as they are unchecked)
		#
		# Condition G = No more requests can be generated at this time based on the available samples and task state (nullified whenever the task state is modified other than to commit generated requests)
		# Condition P = The request pool is empty (nullified if requests are added to the pool)
		# Condition Q = The request queue is empty (nullified if requests from the pool are committed to the queue)
		# Condition B = The request queue does not have enough requests to automatically trigger a full local batch (nullified if requests from the pool are committed to the queue)
		# Condition L = There are no unpushed local batches (nullified whenever a local batch is created)
		# Condition M = No local batch can currently be pushed, either because there are no local batches or because push limits have been reached (nullified whenever a local batch is created or a finished remote batch is processed)
		# Condition R = There are no pushed remote batches that are unfinished (nullified whenever a local batch is pushed)
		# Condition F = There are no pushed remote batches that are finished yet unprocessed (nullified if self.GR.update_remote_batch_state() is internally called)
		# Condition C = [Batch pipeline congestion] No local batches can currently be pushed due to push limits having been reached, but there are at least some configured threshold (>=1) of local batches waiting to be pushed (nullified whenever a local batch is created or a finished remote batch is processed)
		#
		# Condition relations: Q implies B, C implies not[L], C implies M
		# Condition to end a step = E = PBMF(GQ + C)
		# Condition to be completely done = D = GPQBLMRF = ELR (as C implies not[L], note that C must be false in order to potentially be completely done)

		self.step_num += 1
		log.info('-' * 80)
		log.info(f"Step {self.step_num}...")

		while True:

			# Process all finished remote batches (nullifies then ensures condition F, nullifies conditions M and C), and update the task state (nullifies condition G)
			# Requests may internally be auto-retried (if it occurs, temporarily nullifies condition P then re-ensures it and nullifies conditions Q and B)
			self.process_batches()
			if self.GR.only_process:
				log.warning("Only-process mode => Stopping step after having only processed any finished batches")
				all_done = False
				break

			# Push local batches to the server up to the extent of the push limits (ensures condition M, sets condition L if no push limits were reached, nullifies condition R)
			if self.GR.num_unpushed_batches() > 0:
				log.info("Checking whether any local batches can be pushed to the remote server...")
				batch_congestion = self.GR.push_batches()  # C = Returns whether no further local batches can be pushed due to push limits having been reached despite there being at least a certain threshold (>=1) of local batches waiting to be pushed
			else:
				batch_congestion = False

			# Generate requests based on the available samples and task state (sets condition G if generation_done is True) and add them to the request pool (nullifies condition P)
			if batch_congestion:
				generation_done = False  # Condition G = Batch congestion is preventing further requests from being generated, so there may be more once the congestion is over
			else:
				generation_done = self.call_generate_requests()  # Condition G = Returns whether there are no more requests that can be generated right now

			# Commit all pooled requests to the request queue (may also happen intermittently inside generate_requests() call above, ensures condition P, nullifies conditions Q and B)
			# Create local batches out of the requests in the request queue, including a non-full trailing batch if condition G, otherwise some requests may remain in the queue (ensures condition B, ensures condition Q if G, nullifies conditions L, M and C)
			# Push any local batches to the server up to the extent of the push limits (ensures condition M, sets condition L if no push limits were reached, nullifies condition R)
			batch_congestion = self.commit_requests(batch=True, force_batch=generation_done, push=True)  # C = Returns whether no further local batches can be pushed due to push limits having been reached despite there being at least a certain threshold (>=1) of local batches waiting to be pushed

			# Check whether the step and/or entire task is completed (nothing more to do)
			assert self.GR.PQ.pool_len <= 0 and self.GR.num_finished_batches() <= 0  # Assert PF (conditions B and M are less simple to assert, but should also always be true here)
			if (generation_done and self.GR.PQ.queue_len <= 0) or batch_congestion:  # Condition E = PBMF(GQ + C) = GQ + C as conditions P, B, M and F are all guaranteed due to the commands above by just reaching this line
				all_done = (self.GR.num_unpushed_batches() <= 0 and self.GR.num_unfinished_batches() <= 0)  # Condition D = ELR = There are no unpushed local batches and no unfinished remote batches (condition E must be met simply by reaching this line)
				break

		log.info(f"Finished step {self.step_num}")
		return not all_done

	# Call the generate requests implementation
	def call_generate_requests(self) -> bool:
		# Passes on the return value of generate_requests()
		log.info("Generating requests...")
		assert not self.generating
		self.generating = True
		generation_done = self.generate_requests()
		self.generating = False
		log.info(f"Finished generating requests for now => Generation {'DONE' if generation_done else 'ONGOING'}")
		return generation_done

	# Generate requests based on the current task state and add the requests to the GPT requester
	def generate_requests(self) -> bool:
		# Iterate through available samples and generate and add (using self.GR.add_request() / self.GR.add_requests()) GPTRequest instances for work yet to do (i.e. requests that haven't previously already been generated and committed).
		# The task state, and in particular the committed_meta/committed_samples fields thereof (in combination with sample key strings) can be used to determine what work has already been generated, and what work is yet to do.
		# Return a boolean whether there are no more requests that can be generated right now, e.g. because all available samples have been iterated through, and all corresponding requests that can be foreseen for now have been generated.
		# A returned boolean of True MUST mean that if generate_requests() were to immediately be called again (after committing the previously generated requests) then NO further requests would be generated.
		# It must be the case that if generate_requests() is called repeatedly in succession it eventually returns True when there is nothing more that can be generated right now (e.g. at least until some currently in-progress requests finish in the future and potentially allow more requests to then be required).
		# Generated requests must contain enough metadata to allow unambiguous updating of the task state later in commit_cached_request(), i.e. at the very least the sample key(s) associated with the request, as well as any metadata required for later response processing and output file writing.
		# To allow samples to be committed and batched more incrementally (memory/disk consideration), it is permitted to manually call self.commit_requests() at any intermediate time in this method. Otherwise, it can be assumed that self.commit_requests() will be called automatically after this method returns.
		# If calling self.commit_requests() returns that the batch pipeline is currently congested (a certain number of batches are complete and pending but cannot be pushed yet due to thresholds), it is recommended to return early from this method (with return value False as generation is not necessarily done just because congestion occurs) and leave the generation of further requests to future calls of this method.
		raise NotImplementedError

	# Commit generated requests, and optionally batch and push them
	def commit_requests(self, batch: bool = True, force_batch: bool = False, push: bool = True) -> bool:
		# batch = Whether to batch requests after committing them
		# force_batch = Whether to force batching of all requests, i.e. whether to generate a trailing non-full batch with whatever requests are left
		# push = Whether to push batches (if possible) after creating them
		# Returns whether the batch pipeline is currently congested (always False if push=False)

		if self.generating:
			log.info(f"Generated a chunk of {len(self.GR.P)} requests")

		if self.GR.P or any(cached_req is None for cached_req in self.GR.Q):

			log.info("Committing generated requests...")

			with self.GR.commit_requests() as (rstack, cached_reqs):
				if cached_reqs:

					def revert_committed_state(meta: dict[str, Any], samples_all: bool, samples: dict[str, Any]):
						self.T.committed_meta = meta
						if samples_all:
							self.T.committed_samples = samples
						else:
							for sample_key, data in samples.items():
								if data is DELETE:
									self.T.committed_samples.pop(sample_key, None)
								else:
									self.T.committed_samples[sample_key] = data

					DELETE = object()
					if (sample_keys := self.cached_request_keys(cached_reqs)) is None:
						rstack.callback(revert_committed_state, meta_=copy.deepcopy(self.T.committed_meta), samples_all=True, samples=copy.deepcopy(self.T.committed_samples))
					else:
						rstack.callback(revert_committed_state, meta_=copy.deepcopy(self.T.committed_meta), samples_all=False, samples={sample_key: (copy.deepcopy(self.T.committed_samples[sample_key]) if sample_key in self.T.committed_samples else DELETE) for sample_key in sample_keys})

					for cached_req in cached_reqs:
						self.commit_cached_request(cached_req)

					self.validate_state(clean=False)
					self.task.save(rstack=rstack)

			log.info(f"Committed {len(cached_reqs)} generated requests")

		if batch and self.GR.Q:
			log.info(f"Attempting to {'force-' if force_batch else ''}create local batches from {len(self.GR.Q)} available committed requests...")
			num_unpushed_batches = self.GR.batch_requests(force=force_batch)
			if num_unpushed_batches > 0:
				log.info(f"The total number of unpushed local batches is now {num_unpushed_batches}")

		if push and self.GR.num_unpushed_batches() > 0:
			log.info("Checking whether any local batches can be pushed to the remote server...")
			batch_congestion = self.GR.push_batches()
		else:
			batch_congestion = False

		return batch_congestion

	# Update the committed_meta/committed_samples task state to reflect that a particular CachedGPTRequest has been committed
	def commit_cached_request(self, cached_req: gpt_requester.CachedGPTRequest):
		# cached_req = CachedGPTRequest that has been committed
		raise NotImplementedError

	# Extract from a list of CachedGPTRequest's a set of sample keys that is enough to cover all possible changes to the committed_samples task state when supplying these CachedGPTRequest's to commit_cached_request()
	def cached_request_keys(self, cached_reqs: list[gpt_requester.CachedGPTRequest]) -> Optional[set[str]]:  # noqa
		# cached_reqs = List of CachedGPTRequest's to extract all relevant sample key strings from
		# Return a set of sample key strings (the set or superset of sample key strings that will be modified by commit_cached_request()), or None (caller must assume all sample keys could be modified)
		return None

	# Make and process a series of direct requests
	def direct_requests(self, reqs: Union[Iterable[gpt_requester.GPTRequest], gpt_requester.GPTRequest]):
		# reqs = The requests to make and process using the direct API
		# This method performs direct API calls and updates the task/requester state
		for rstack, result in self.GR.direct_requests(reqs=reqs):
			if self.process_batch_result(result=result, rstack=rstack):
				self.validate_state(clean=False)
				self.output.validate()
				self.task.save(rstack=rstack)
				self.output.save(rstack=rstack)

	# Process and clean up after any finished remote batches
	def process_batches(self) -> int:
		# Returns the current number of unfinished remote batches (after the remote batch status updates)
		# This method checks the remote for updated batch statuses, collects the results of any finished batches, updates the task/requester state, and cleans up that batch (also from the remote), moving the corresponding BatchState from batches to batch_history.
		for rstack, result in self.GR.process_batches():
			if self.process_batch_result(result=result, rstack=rstack):
				self.validate_state(clean=False)
				self.output.validate()
				self.task.save(rstack=rstack)
				self.output.save(rstack=rstack)
		assert self.GR.num_finished_batches() <= 0  # Condition F
		return self.GR.num_unfinished_batches()

	# Process a batch result and accordingly update the task state and output files (must be a perfectly reversible operation managed by the RevertStack rstack)
	def process_batch_result(self, result: gpt_requester.BatchResult, rstack: utils.RevertStack) -> bool:
		# result => The result of a batch, to be processed and used to update the task output file and task state
		# rstack => A RevertStack so that the actions of this method are perfectly reversible in the case of an exception
		# Returns whether the task state (self.T) or output (self.D) was modified and thus that both need to be saved (failed requests should not affect the task output file at all)
		#
		# The final batch state---including the final remote batch status (result.batch.remote.batch: openai.type.Batch), API metrics (result.batch.metrics: APIMetrics), and true cost (result.batch.true_tokens_cost: TokensCost)---is available in result.batch (BatchState).
		# If the batch encountered any general/global errors then these are listed in result.errors (list[openai.types.BatchError])---however, there may nonetheless be valid responses even if there are such errors, so best to just immediately check the responses instead if that's all that counts.
		# The main results/responses to process are the values of the result.info dict, which maps request IDs (int, returned from self.GR.add_request() / self.GR.add_requests()) to ResultInfo instances.
		# The following information is available for each ResultInfo instance 'info':
		#   - The input request payload (info.req_payload: dict[str, Any]) and metadata (info.req_info.meta: Optional[dict[str, Any]])
		#   - If a response was received for the request (info.resp_info is not None), the response in Python class/parsed format (info.resp_info.payload: openai.types.chat.ChatCompletion/ParsedChatCompletion or similar depending on auto-parse and endpoint)
		#   - If an error occurred with the request and/or response (info.err_info is not None), the error (info.err_info: ErrorInfo) => At least one of info.err_info and info.resp_info will always be non-None
		#   - If warnings occurred while processing the response, the warnings (info.warn_infos: list[ErrorInfo])---warnings occur for example if multiple completion choices are requested and some choices fail somehow while others don't
		#   - Whether the request will be retried (info.retry: bool) and whether the current result counts towards the retry number (info.retry_counts: bool / e.g. batch cancellation or expiry by default does not count)
		#   - The default (on entering this method) value of info.retry is never True if there is no error present (info.err_info is None)
		#   - The field info.retry can be MODIFIED in this method to set whether the request will get retried (e.g. because of a task-specific parsing or value failure)
		#   - Theoretically, info.req_payload, info.req_info.meta, info.retry_counts and info.req_info.retry_num can also be MODIFIED to affect/tweak the retry, but is not recommended in general
		# Statistics like the remote batch completion duration, request pass ratio, and number of requests that were successful, warned, errored, cancelled, expired, etc, can be found in result.stats (ResultStats).
		raise NotImplementedError
# EOF
