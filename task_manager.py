# Task manager (wraps a GPT requester to include sample generation and output file management)

# Imports
from __future__ import annotations
import os
import re
import abc
import copy
import json
import math
import types
import argparse
import functools
import itertools
import contextlib
import collections
import dataclasses
from typing import Any, Optional, Self, Type, Callable, Generic, Iterable, Union, Counter, TypeVar
from .logger import log
from . import gpt_requester, utils

# Type variables
T = TypeVar('T')
DataclassT = TypeVar('DataclassT')

#
# Task state file
#

# Task state class
@dataclasses.dataclass
class TaskState:

	version: int = 1                                                             # Data format version number
	meta: dict[str, Any] = dataclasses.field(default_factory=dict)               # Task-level metadata (arbitrary JSON-compatible key-value store, e.g. can be used to store parameters that should/must not change throughout an entire task, even if the task is resumed with different configuration arguments, TaskManager never modifies the meta for as long as the task manager is entered)
	committed_meta: dict[str, Any] = dataclasses.field(default_factory=dict)     # Task-level information about the samples committed so far (arbitrary JSON-compatible key-value store)
	committed_samples: dict[str, Any] = dataclasses.field(default_factory=dict)  # Sample-level information about the samples committed so far (maps sample keys to corresponding JSON-compatible data)
	responded_meta: dict[str, Any] = dataclasses.field(default_factory=dict)     # Task-level information about the samples that have received a response so far (arbitrary JSON-compatible key-value store)
	failed_samples: dict[str, Any] = dataclasses.field(default_factory=dict)     # Sample-level information about the committed samples that have received a response so far and failed (maps sample keys to corresponding JSON-compatible data, keys MUST be a non-strict subset of committed_samples at all times)
	succeeded_samples: dict[str, Any] = dataclasses.field(default_factory=dict)  # Sample-level information about the committed samples that have received a response so far and succeeded (maps sample keys to corresponding JSON-compatible data, keys MUST be a non-strict subset of committed_samples at all times)

	def summary(self) -> dict[str, Any]:
		# Returns a flat JSON-compatible summary of the data contained in the class

		meta_summary = {}
		for meta, base_key in (
			(self.meta, 'TaskMeta/'),
			(self.committed_meta, 'Task/committed_'),
			(self.responded_meta, 'Task/responded_'),
		):
			for key, value in itertools.islice(meta.items(), 100):  # For safety, only include up to 100 items in the summary (we never want to have hundreds or thousands here, so rather omit some items than spam the summary and potentially slow down wandb)
				if isinstance(value, (bool, int, float, str, types.NoneType)):
					meta_summary[base_key + key] = value

		num_committed = len(self.committed_samples)
		num_pending = len(self.committed_samples.keys() - self.failed_samples.keys() - self.succeeded_samples.keys())
		both_keys = self.failed_samples.keys() & self.succeeded_samples.keys()

		return {
			'Task/samples_committed': num_committed,                                   # Committed at least once
			'Task/samples_pending': num_pending,                                       # No response received yet at all
			'Task/samples_responded': num_committed - num_pending,                     # Responded at least once, but not necessarily yet as many times as committed
			'Task/samples_succeeded': len(self.succeeded_samples.keys() - both_keys),  # All responses so far were successes
			'Task/samples_failed': len(self.failed_samples.keys() - both_keys),        # All responses so far were failures
			'Task/samples_mixed_success': len(both_keys),                              # Some responses were successes, some were failures
			**meta_summary,
		}

# Task state file class
class TaskStateFile:

	state: Optional[TaskState]

	def __init__(self, path: str, reinit_meta: bool, init_meta: Optional[dict[str, Any]], dryrun: bool, W: Optional[utils.WandbRun] = None):
		# path = Path to the JSON task state file to load/save/manage (nominally *.json extension)
		# reinit_meta = Whether to force a reinitialization of the meta field even if the task state file already exists
		# init_meta = Value to initialize the meta field with if the task state file is newly created (deep copy on create)
		# dryrun = Whether to prevent any saving of state (dry run mode)
		# W = Wandb run holder
		self.path = os.path.abspath(path)
		self.name = os.path.basename(self.path)
		log.info(f"Task state file: {self.path}")
		self.reinit_meta = reinit_meta
		self.init_meta = init_meta if init_meta is not None else {}
		self.dryrun = dryrun
		self.W = W if W is not None else utils.WandbRun()
		self._enter_stack = contextlib.ExitStack()
		self.state = None

	# noinspection PyUnusedLocal
	def clear_reinit_meta(self, exc_type, exc_val, exc_tb) -> bool:
		if exc_type is None:
			self.reinit_meta = False
		return False

	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicLogRevertStack(W=self.W) as rstack:
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
		self.log_summary(rstack=rstack)

	def save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible saving of the task state file
		if self.dryrun:
			log.warning(f"{gpt_requester.DRYRUN}Did not save task state file with {len(self.state.committed_samples)} committed, {len(self.state.failed_samples)} failed, and {len(self.state.succeeded_samples)} succeeded sample keys")
		else:
			with utils.SafeOpenForWrite(path=self.path, rstack=rstack) as file:
				utils.json_from_dataclass(obj=self.state, file=file)
				file_size = utils.get_file_size(file)
			log.info(f"Saved task state file with {len(self.state.committed_samples)} committed, {len(self.state.failed_samples)} failed, and {len(self.state.succeeded_samples)} succeeded sample keys ({utils.format_size_iec(file_size)})")
		self.log_summary(rstack=rstack)

	def unload(self):
		self.state = None

	def log_summary(self, rstack: utils.RevertStack):
		# rstack = Possible LogRevertStack (subclass of RevertStack) to use for logging
		if isinstance(rstack, utils.LogRevertStack):
			rstack.log(self.state.summary())

#
# Task output file
#

# Task output file class
class TaskOutputFile(abc.ABC):

	def __init__(self, path_base: str, dryrun: bool, W: Optional[utils.WandbRun] = None):
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output
		# dryrun = Whether to prevent any saving of output (dry run mode)
		# W = Wandb run holder
		self.path_base = os.path.abspath(path_base)
		self.dryrun = dryrun
		self.W = W if W is not None else utils.WandbRun()
		self.data = None
		self.written = False

	def __enter__(self) -> Self:
		# See _enter()
		self.written = False
		return self._enter()

	@abc.abstractmethod
	def _enter(self) -> Self:
		# Load/create the task output file and set/initialize self.data (must be mutable or None)
		raise NotImplementedError

	@abc.abstractmethod
	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		# Unload the task output file (don't automatically just save the task output file here - it is the responsibility of the user of the class to explicitly call save() when required)
		raise NotImplementedError

	def validate(self):
		# Assuming the class is entered, perform any possible validations on the data and raise a ValueError for any failures
		pass

	def create(self, rstack: utils.RevertStack):
		# See _create()
		self.written = True
		self._create(rstack=rstack)

	@abc.abstractmethod
	def _create(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible creation of the task output file
		# Initialize the task output file on disk and in memory
		raise NotImplementedError

	def reset(self, rstack: utils.RevertStack):
		# See _reset()
		self.written = True
		self._reset(rstack=rstack)

	@abc.abstractmethod
	def _reset(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible resetting of the task output file
		# Reset the task output file to the state it is in right after creation (don't necessarily assume that any self.data is currently already created/loaded)
		raise NotImplementedError

	@abc.abstractmethod
	def load(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible loading of the task output file
		# Load the task output file to memory
		raise NotImplementedError

	def save(self, rstack: utils.RevertStack):
		# See _save()
		self.written = True
		self._save(rstack=rstack)

	@abc.abstractmethod
	def _save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible saving of the task output file
		# Save the current memory state of the task output file to disk
		# It is permitted to make changes to self.data (e.g. sorting keys of a dictionary or so) just prior to saving, as long as it is reversible (rstack)
		raise NotImplementedError

	def unload(self):
		# Unload the task output file
		self.data = None

	def log_summary(self, *args: dict[str, Any], rstack: utils.RevertStack, flush: bool = False, **kwargs: Any):
		# rstack = Possible LogRevertStack (subclass of RevertStack) to use for logging
		# args, flush, kwargs = Arguments to pass on to rstack.log() if rstack is a LogRevertStack
		if isinstance(rstack, utils.LogRevertStack):
			rstack.log(*args, flush=flush, **kwargs)

# Dataclass output file class
class DataclassOutputFile(TaskOutputFile, Generic[DataclassT]):

	data: Optional[DataclassT]  # TaskOutputFile: Data backed by the output file (while the class is in the entered state)

	@classmethod
	def read(cls, path_base: str, W: Optional[utils.WandbRun] = None, data_cls: Optional[Type[DataclassT]] = None) -> Self:
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output
		# W = Wandb run holder
		# data_cls = Dataclass type to use (must be instantiatable without arguments, None = Assume cls.Dataclass exists and use it)
		# Returns a new instance of the class in read-only mode (raises an exception if load() fails on enter, or a save() is attempted)
		return cls(path_base=path_base, dryrun=True, W=W, data_cls=data_cls, read_only=True)

	@classmethod
	def output_factory(cls, data_cls: Optional[Type[DataclassT]] = None) -> Callable[[str, bool, Optional[utils.WandbRun]], DataclassOutputFile[DataclassT]]:
		# data_cls = Dataclass type to use (must be instantiatable without arguments, None = Assume cls.Dataclass exists and use it)
		# Returns a (not read-only mode) factory function suitable for passing as the output_factory argument of the TaskManager class
		return functools.partial(cls, data_cls=data_cls, read_only=False)

	def __init__(self, path_base: str, dryrun: bool, W: Optional[utils.WandbRun] = None, *, data_cls: Optional[Type[DataclassT]], read_only: bool):
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output (extension of .json will automatically be added)
		# dryrun = Whether to prevent any saving of output and just log what would have happened instead (dry run mode)
		# W = Wandb run holder
		# data_cls = Dataclass type to use (must be instantiatable without arguments, None = Assume cls.Dataclass exists and use it)
		# read_only = Whether to use read-only mode (raises an exception if load() fails on enter, or a save() is attempted)
		super().__init__(path_base=path_base, dryrun=dryrun, W=W)
		try:
			self.data_cls = data_cls if data_cls is not None else getattr(type(self), 'Dataclass')
		except AttributeError:
			raise ValueError("If no explicit dataclass type is provided via data_cls=X, then this class must be a subclass with a defined 'Dataclass' class attribute")
		self.read_only = read_only
		self.path = f'{self.path_base}.json'
		self.name = os.path.basename(self.path)
		log.info(f"Task output file: {self.path}")
		self._enter_stack = contextlib.ExitStack()

	def _enter(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicLogRevertStack(W=self.W) as rstack:
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

	def _create(self, rstack: utils.RevertStack):
		rstack.callback(setattr, self, 'data', self.data)
		self.data = self.data_cls()
		self._save(rstack=rstack)

	def _reset(self, rstack: utils.RevertStack):
		self._create(rstack=rstack)

	def load(self, rstack: utils.RevertStack):
		rstack.callback(setattr, self, 'data', self.data)
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.data = utils.dataclass_from_json(cls=self.data_cls, json_data=file)
		log.info(f"Loaded task output file with {self.status_str()} ({utils.format_size_iec(file_size)})")
		self.log_summary(rstack=rstack)

	def pre_save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use to make any changes to the data reversible
		# This method can be overridden to perform changes to self.data immediately prior to each save (e.g. sorting keys of a dictionary or so)
		pass

	def _save(self, rstack: utils.RevertStack):
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
		self.log_summary(rstack=rstack)

	def log_summary(self, *args: dict[str, Any], rstack: utils.RevertStack, flush: bool = False, **kwargs: Any):
		# It is encouraged to override this method to log a more useful summary (as subclasses know what DataclassT actually is)
		super().log_summary({
			'Output/num_data_fields': len(dataclasses.fields(self.data)),
		}, *args, rstack=rstack, flush=flush, **kwargs)

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
	def read(cls, path_base: str, W: Optional[utils.WandbRun] = None, data_cls: Optional[Type[DataclassT]] = None) -> Self:
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output
		# W = Wandb run holder
		# data_cls = Dataclass type to use for each list entry (None = Assume cls.Dataclass exists and use it)
		# Returns a new instance of the class in read-only mode (loads all entries from all output files, raises an exception if load() fails on enter or a save() is attempted)
		return cls(path_base=path_base, dryrun=True, W=W, data_cls=data_cls, read_only=True, max_entries=0, max_size=0)

	@classmethod
	def output_factory(cls, data_cls: Optional[Type[DataclassT]] = None, max_entries: int = 0, max_size: int = 0) -> Callable[[str, bool, Optional[utils.WandbRun]], DataclassListOutputFile[DataclassT]]:
		# data_cls = Dataclass type to use for each list entry (None = Assume cls.Dataclass exists and use it)
		# max_entries = Maximum number of entries to save per output file chunk (<=0 = No maximum)
		# max_size = Maximum file size in bytes per output file chunk (<=0 = No maximum)
		# Returns a (not read-only mode) factory function suitable for passing as the output_factory argument of the TaskManager class
		return functools.partial(cls, data_cls=data_cls, read_only=False, max_entries=max_entries, max_size=max_size)

	def __init__(self, path_base: str, dryrun: bool, W: Optional[utils.WandbRun] = None, *, data_cls: Optional[Type[DataclassT]], read_only: bool, max_entries: int = 0, max_size: int = 0):
		# path_base = Required output file path base, e.g. /path/to/NAME_PREFIX_output (suffix of .jsonl or _partXofY.jsonl will automatically be added as appropriate)
		# dryrun = Whether to prevent any saving of output and just log what would have happened instead (dry run mode)
		# W = Wandb run holder
		# data_cls = Dataclass type to use for each list entry (None = Assume cls.Dataclass exists and use it)
		# read_only = Whether to use read-only mode (loads all entries from all output files, raises an exception if load() fails on enter or a save() is attempted)
		# max_entries = Maximum number of entries to save per output file chunk (<=0 = No maximum, not relevant in read-only mode)
		# max_size = Maximum file size in bytes per output file chunk (<=0 = No maximum, not relevant in read-only mode)
		super().__init__(path_base=path_base, dryrun=dryrun, W=W)
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

	def _enter(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicLogRevertStack(W=self.W) as rstack:
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

	def _create(self, rstack: utils.RevertStack):
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
		self.log_summary(rstack=rstack)

	def _reset(self, rstack: utils.RevertStack) -> list[str]:
		# Extends the base class signature with a return value => Returns a list of the temporary unlink backup paths
		backup_paths = []
		if self.data is not None and not self.dryrun:
			for path in self.data.paths:
				backup_path = utils.safe_unlink(path=path, rstack=rstack)
				if backup_path is not None:
					backup_paths.append(backup_path)
		self._create(rstack=rstack)
		return backup_paths

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
		self.log_summary(rstack=rstack)

	def pre_save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use to make any changes to the data entries reversible
		# This method can be overridden to perform changes to self.data.entries immediately prior to them being saved and cleared (e.g. sorting keys of a dictionary or so)
		pass

	def _save(self, rstack: utils.RevertStack):

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

		added_entry_size = 0
		last_lines = []

		def save_last_lines():
			nonlocal added_entry_size
			if not self.dryrun:
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
					if not self.dryrun:
						os.replace(src=path, dst=correct_path)
						rstack.callback(os.replace, src=correct_path, dst=path)
					self.data.paths[p] = correct_path

		if self.dryrun:
			log.warning(f"{gpt_requester.DRYRUN}Did not append {len(self.data.entries)} entries to the now {len(self.data.paths)} task output file(s) (added {utils.format_size_iec(added_entry_size)})")
		else:
			log.info(f"Appended {len(self.data.entries)} entries to the now {len(self.data.paths)} task output file(s) (added {utils.format_size_iec(added_entry_size)})")

		self.data.entries.clear()
		self.log_summary(rstack=rstack)

	def log_summary(self, *args: dict[str, Any], rstack: utils.RevertStack, flush: bool = False, **kwargs: Any):
		super().log_summary({
			'Output/num_files': len(self.data.paths),
			'Output/max_file_entries': self.max_entries,
			'Output/last_file_entries': self.data.last_entries,
			'Output/max_file_size_mb': self.max_size / 1048576,
			'Output/last_file_size_mb': self.data.last_size / 1048576,
		}, *args, rstack=rstack, flush=flush, **kwargs)

	def entries(self) -> Iterable[DataclassT]:
		for path in self.data.paths:
			with open(path, 'r', encoding='utf-8') as file:
				for line in file:
					yield utils.dataclass_from_json(cls=self.data_cls, json_data=line)

	def all_entries(self) -> list[DataclassT]:
		return list(self.entries())

	def rewrite(self, rstack: utils.RevertStack) -> Iterable[DataclassT]:
		# While iterating through this generator with a for-loop, add any desired new contents to self.data.entries (for example, in order to simply filter the output file data, append the yielded entry to self.data.entries only if a particular conditional expression is true)
		# Note that internally a reset occurs, so afterwards self.data points to a new instance
		self.written = True
		if self.read_only:
			raise RuntimeError("Cannot rewrite dataclass list output file in read-only mode")
		data_paths = self.data.paths
		backup_paths = self._reset(rstack=rstack)
		for path in (data_paths if self.dryrun else backup_paths):
			with open(path, 'r', encoding='utf-8') as file:
				for line in file:
					yield utils.dataclass_from_json(cls=self.data_cls, json_data=line)
			self.validate()
			self._save(rstack=rstack)

#
# Task manager class
#

# Task manager class
@utils.init_kwargs
class TaskManager:

	# This abstract base class manages a gpt_requester.GPTRequester instance to complete a certain custom task. Task progress is stored in a task state file as well as a task output file.
	# This is in addition to the files stored by gpt_requester.GPTRequester, and the task files are also only updated when the gpt_requester.GPTRequester has its lock file locked.
	#
	# The task state file is of the format given by the TaskState class above => REVIEW the TaskState class above before continuing (e.g. defines committed_samples, failed_samples, succeeded_samples)
	# The task output file is of type TaskOutputFile (without any particular mandated format) => Example implementations based on dataclasses are DataclassOutputFile and DataclassListOutputFile
	#
	# In order to use this class, subclass it for a particular task and implement/override:
	#   - __init__(cfg: utils.Config) => Customize a call to super().__init__(..., **gpt_requester_kwargs) based on an attribute-based cfg (coming from either Python argparse or hydra, see below) where gpt_requester_kwargs comes from gpt_requester.GPTRequester.get_kwargs(cfg)
	#   - on_task_enter()             => [Optional] Perform any required custom actions during entering of the task manager (called once task is entered and self.T is available, but before GPT requester is entered)
	#   - on_task_exit()              => [Optional] Perform any required custom actions during exiting the task manager (called if on_task_enter() was executed to completion during entering)
	#   - wipe_unfinished()           => Wipe any unfinished (and optionally also failed) requests/samples from the in-memory task state
	#   - validate_state()            => Validate that there are no obvious issues with the current state (remember to call super())
	#   - generate_requests()         => Implement request generation based on current task state
	#   - commit_cached_request()     => Update the committed_meta/committed_samples task state to reflect that a particular request has been committed
	#   - cached_request_keys()       => Extract from a list of requests a set of sample keys that is enough to cover all possible changes to the committed_samples task state when later supplying these requests to commit_cached_request()
	#   - process_batch_result()      => Process a batch result and accordingly update the task state and output files (must be a perfectly reversible operation)
	#   - postprocess_output_data()   => [Optional] Perform any required post-processing of the task output data (called only once the entire task completes)
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
	#  - Given just the task state, it should be possible to reduce the record of committed/responded samples to just those for which a succeeded response was received so far (failed samples are wiped, the task output needs to be able to be adjusted accordingly if it is potentially affected even though no succeeded samples are touched)
	#
	# The init_meta argument to __init__ specifies parameter values that should always remain fixed throughout a task, even across multiple runs (this behavior can be manually overridden using reinit_meta).

	# Construct a task manager to make use of the OpenAI Batch API to process samples
	def __init__(
		self,
		task_dir: str,                                                          # Path of the task working directory to use (will be used for automatically managed lock, state, requests, batch, task state, and output files)
		name_prefix: str,                                                       # Name prefix to use for all files created in the task working directory (e.g. 'my_task')
		output_factory: Callable[[str, bool, utils.WandbRun], TaskOutputFile],  # Factory callable to create the required task output file instance (str argument is the required output file path base, e.g. /path/to/NAME_PREFIX_output, bool argument is whether dry run mode is active, utils.WandbRun is a wandb run holder)
		init_meta: Optional[dict[str, Any]],                                    # Value to initialize the task state meta field with if the task state file is newly created (deep copy on create)
		*,                                                                      # Keyword arguments only beyond here

		run: bool = True,                                                       # Whether to execute steps when the task manager is run, or just show the status and return (e.g. run=False is useful in combination with wipe_*)
		postprocess_output: str = 'if_written',                                 # Whether to allow/perform custom post-processing of the task output data if a run totally completes the task (options: never, if_written, always, where if_written means only if the task output file was written to during the run)
		wipe_failed: bool = False,                                              # CAUTION: Wipe and forget all failed samples from the task state (implies wipe_requests, consider running the task with only_process=True prior to wiping)
		reinit_meta: bool = False,                                              # CAUTION: Whether to force a reinitialization of the task state meta field even if the task state file already exists (normally the task state meta field is only initialized once at the beginning of a task and remains fixed after that across all future runs)

		**gpt_requester_kwargs,                                                 # Keyword arguments to be passed on to the internal GPTRequester instance
	):

		self.run_flag = run
		self.postprocess_output = postprocess_output
		if self.postprocess_output not in ('never', 'if_written', 'always'):
			raise ValueError(f"Invalid post-process output mode: {self.postprocess_output}")
		self.wipe_failed = wipe_failed
		if self.wipe_failed:
			gpt_requester_kwargs['wipe_requests'] = True
		if gpt_requester_kwargs.get('wandb', None) is None:
			gpt_requester_kwargs['wandb'] = not gpt_requester_kwargs.get('dryrun', False)

		self.GR = gpt_requester.GPTRequester(working_dir=task_dir, name_prefix=name_prefix, **gpt_requester_kwargs)
		self.task = TaskStateFile(path=os.path.join(self.GR.working_dir, f"{self.GR.name_prefix}_task.json"), reinit_meta=reinit_meta, init_meta=init_meta, dryrun=self.GR.dryrun, W=self.GR.W)
		self.output = output_factory(os.path.join(self.GR.working_dir, f"{self.GR.name_prefix}_output"), self.GR.dryrun, self.GR.W)  # Arguments: path_base (str), dryrun (bool), W (utils.WandbRun)

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
		add_argument(name='postprocess_output', metavar='MODE', help="Whether to allow/perform custom post-processing of the task output data if a run totally completes the task (options: never, if_written, always, where if_written means only if the task output file was written to during the run)")
		add_argument(name='wipe_failed', help="CAUTION: Wipe and forget all failed samples from the task state (implies wipe_requests, consider running the task with --only_process prior to wiping)")
		add_argument(name='reinit_meta', help="CAUTION: Whether to force a reinitialization of the task state meta field even if the task state file already exists (normally the task state meta field is only initialized once at the beginning of a task and remains fixed after that across all future runs)")

		return group

	# Run the task manager to completion (or as far as possible)
	def run(self) -> bool:
		# Returns whether the task manager is complete (all done)

		log.info('\u2550' * 120)
		log.info("Running task manager...")

		with self:

			log.info('\xB7' * 60)
			self.log_status()
			self.GR.log_status()

			all_done = False
			if self.run_flag:
				while self.step():  # Returns True exactly if condition E*not[D] = PBVMF(GQ + C)(not[L] + not[R]), i.e. only if condition F is satisfied => There are no pushed remote batches that are finished yet unprocessed
					if self.GR.dryrun:
						log.warning(f"{gpt_requester.DRYRUN}Stopping incomplete task manager as it is a dry run")
						break
					elif self.GR.num_unfinished_batches() <= 0:  # If condition R... (i.e. PBVMFR and not[L] and (GQ or C) => There are local unpushed batches but no remote batches and everything else is all done, except possibly batch pipeline congestion)
						log.warning("Stopping incomplete task manager as a step did not result in unfinished pushed remote batches")
						break
					else:  # Else if condition not[R]F...
						self.GR.wait_for_batches()  # Waits (if not a dry run) until condition R + not[F] (nullifies condition F) => When this returns there must be at least one finished yet unprocessed remote batch, or no unfinished and/or unprocessed remote batches at all
				else:  # Natural exit of while loop means self.step() returned False, indicating there is no more work left to do
					all_done = True
			else:
				log.warning("Not running the task manager due to 'run' flag being False")

			log.info('\u2550' * 120)
			self.log_status()
			self.GR.log_status()

			if self.postprocess_output == 'always' or (self.postprocess_output == 'if_written' and self.output.written):
				log.info('\xB7' * 60)
				log.info("Performing any required post-processing of the task output data...")
				self.postprocess_output_data()
				log.info("Finished all required post-processing of the task output data")

		log.info('\xB7' * 60)
		log.info(f"Finished running task manager ({'all done' if all_done else 'work left to do'})")

		return all_done

	# Callback that can be used to perform custom actions during entering (called once task is entered and self.T is available, but before GPT requester is entered)
	def on_task_enter(self):
		pass

	# Callback that is called on exit if on_task_enter() executed to completion on enter
	def on_task_exit(self):
		pass

	# Custom extra wandb configuration parameters
	@property
	def extra_wandb_configs(self) -> dict[str, Any]:
		return dict(
			run=self.run_flag,
			reinit_meta=self.task.reinit_meta,
		)

	# Enter method for the required use of TaskManager as a context manager
	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			enter_stack.enter_context(self.GR.lock)
			enter_stack.enter_context(self.GR.configure_wandb((self.GR, self.GR.extra_wandb_configs), (self, self.extra_wandb_configs)))
			wipe_requests = self.GR.wipe_requests
			wipe_task = self.GR.wipe_task
			enter_stack.callback(self.on_exit)
			enter_stack.enter_context(self.task)
			self.T = self.task.state
			self.on_task_enter()
			enter_stack.callback(self.on_task_exit)
			enter_stack.enter_context(self.GR)
			self.step_num = 0
			self.generating = False
			self.validate_state(clean=False)
			log.info('\xB7' * 60)
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
				log.info('\xB7' * 60)
				log.warning("Wiping complete task output and state...")
				with utils.LogRevertStack(W=self.GR.W) as rstack:
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
				log.info('\xB7' * 60)
				log.warning(f"Wiping unfinished{' and failed' if wipe_failed else ''} requests/samples...")
				with utils.LogRevertStack(W=self.GR.W) as rstack:
					save_output = self.wipe_unfinished(wipe_failed=wipe_failed, rstack=rstack)
					self.validate_state(clean=True)
					if save_output:
						self.output.validate()
					self.task.save(rstack=rstack)
					if save_output:
						self.output.save(rstack=rstack)
				log.warning(f"Wiped unfinished{' and failed' if wipe_failed else ''} requests/samples")

	# Wipe any unfinished (and optionally also failed) requests/samples from the in-memory task state (self.T) and possibly the task output (self.D)
	def wipe_unfinished(self, wipe_failed: bool, rstack: utils.RevertStack) -> bool:
		# wipe_failed = Whether in addition to the committed-yet-unfinished samples to also wipe the failed samples (i.e. give them another chance)
		# rstack = RevertStack to use in case one is needed (the entire wipe operation does not need to be completely revertible, but some parts of it might be in order to ensure no unintentional data loss)
		# Returns whether the task output should be validated and saved because it was modified
		# The implementation should update the task state (self.T), specifically the committed_samples/committed_meta fields, as well as the failed_samples/responded_meta fields if wipe_failed
		# The task output should only potentially need to be adjusted if wipe_failed=True and the presence of failed samples implicitly affects the task output (e.g. due to a maximum number of total succeeded or failed opinions being allowed per sample before writing to the task output, or something like that)
		# If not wipe_failed, this method should reduce the record of committed samples in self.T to just those committed samples that are responsible for the responses received so far (succeeded or failed)
		# If wipe_failed, this method should reduce the record of committed/responded samples in self.T to just those committed samples that are responsible for the succeeded samples (failed samples are also cleared)
		# The task state after this method finishes must pass self.validate_state(clean=True)
		raise NotImplementedError

	# Validate that there are no obvious issues with the current state
	def validate_state(self, *, clean: bool):
		# clean = Whether the current state should be without any unfinished requests/samples (all committed samples should have a response already)
		# Note: This method can be overridden to sanity check task-specific task state conditions (remember to call this implementation via super() though)
		# Note: The task output may not be loaded at this point
		if not self.T.committed_samples.keys() >= self.T.failed_samples.keys():
			raise ValueError(f"Unexpected failed yet not committed samples: {sorted(self.T.failed_samples.keys() - self.T.committed_samples.keys())}")
		if not self.T.committed_samples.keys() >= self.T.succeeded_samples.keys():
			raise ValueError(f"Unexpected succeeded yet not committed samples: {sorted(self.T.succeeded_samples.keys() - self.T.committed_samples.keys())}")
		if clean:
			responded_sample_keys = self.T.succeeded_samples.keys() | self.T.failed_samples.keys()
			if self.T.committed_samples.keys() != responded_sample_keys:
				raise ValueError(f"Unexpected committed relative to responded samples in clean case, with disagreement in the keys: {sorted(self.T.committed_samples.keys() ^ responded_sample_keys)}")

	# Log the current task manager status
	def log_status(self):
		num_committed = len(self.T.committed_samples)
		num_pending = len(self.T.committed_samples.keys() - self.T.failed_samples.keys() - self.T.succeeded_samples.keys())
		both_keys = self.T.failed_samples.keys() & self.T.succeeded_samples.keys()
		log.info(f"There are {num_committed} committed SAMPLES ({num_pending} pending + {num_committed - num_pending} responded at least once), and {len(self.T.succeeded_samples.keys() - both_keys)} succeeded + {len(self.T.failed_samples.keys() - both_keys)} failed + {len(both_keys)} both SAMPLES")

	# Execute a single non-blocking step of the task manager (generates and processes as much as is currently possible without waiting)
	def step(self) -> bool:
		# Returns whether the task manager has work left to do
		#
		# Assumption: The available samples, on the basis of which generate_requests() generates requests, are fixed for the entire session (duration that a task manager is entered), e.g. for an entire call to run()
		# Assumption: When finished remote batches are processed, they do not have any direct influence on the push limits anymore, and thus on whether the batch pipeline is congested or not
		# Assumption: Local batches are mutually exclusively either 'pushable' (pushed to the remote server and executed via the batch API) or 'unpushable' (executed via the direct API), and this never changes for any particular batch during a session
		# Assumption: The task/session/remote/batch/push limits never change during a session
		#
		# Conditions are boolean variables that are true if the condition is definitely true, and false if something occurs (nullification) that could POSSIBLY make the condition untrue, and rechecking is needed to have it set to true again (all conditions start as untrue as they are unchecked)
		#
		# Condition G = No more requests can be generated at this time based on the available samples and task state (nullified whenever the task state is modified other than to simply commit generated requests)
		# Condition P = The request pool is empty (nullified if requests are added to the pool)
		# Condition Q = The request queue is empty (nullified if requests from the pool are committed to the queue)
		# Condition B = The request queue does not have enough requests to automatically trigger a full local batch (nullified if requests from the pool are committed to the queue)
		# Condition L = There are no unpushed local batches, whether pushable or unpushable (nullified whenever a local batch is created)
		# Condition V = Either there are no unpushable batches, or no single direct request should be made anymore for the remainder of the run/session due to session/task limits (nullified whenever a local batch is created)
		# Condition M = No pushable local batch can currently be pushed, either because there are no pushable local batches or because push limits have been reached (nullified whenever a local batch is created, or a finished remote batch is processed)
		# Condition R = There are no pushed remote batches that are unfinished (nullified whenever a local batch is pushed)
		# Condition F = There are no pushed remote batches that are finished yet unprocessed, or it is a dry run (nullified if self.GR.update_remote_batch_state() is internally called)
		# Condition C = [Batch pipeline congestion] Condition M, and there are at least some configured threshold (>=1) of (pushable or unpushable) local batches available (nullified whenever a local batch is created, a finished remote batch is processed, or an unpushable batch is processed)
		#
		# Condition relations: Q implies B, C implies not[L], C implies M
		# Condition for batch pipeline congestion = C = M * "number of unpushed local batches >= threshold" (where we know that threshold >= 1)
		# Condition to end a step = E = PBVMF(GQ + C)
		# Condition to be completely done = D = GPQBLVMRF = ELR (as C implies not[L], note that C must be false in order to potentially be completely done)

		self.step_num += 1
		log.info('\u2550' * 120)
		log.info(f"Step {self.step_num}...")
		self.GR.W.log(data={'Task/run_step': self.step_num})

		while True:

			# Process all finished remote batches (nullifies then ensures condition F, nullifies conditions M and C), and update the task state (nullifies condition G)
			# Requests may internally be auto-retried (if it occurs, temporarily nullifies condition P then re-ensures it and nullifies conditions Q and B)
			self.process_batches()
			if self.GR.only_process:
				log.warning("Only-process mode => Stopping step after having only processed any finished batches")
				all_done = False
				break

			# Push pushable local batches to the server up to the extent of the push limits (ensures condition M, potentially sets conditions L, nullifies condition R)
			self.push_batches()

			# Process all unpushable local batches up to the extent possible with the direct limits (ensures condition V, potentially sets condition L, nullifies condition C), and update the task state (nullifies condition G)
			num_unpushable_batches, direct_limits_reached = self.process_unpushable_batches()  # Returns how many unpushable batches are still left after processing (will only ever be non-zero if direct_limits_reached, note that condition V = num_unpushable_batches <= 0 or direct_limits_reached), and whether some direct limits were reached

			# Determine whether the batch pipeline is congested
			batch_congestion = (self.GR.num_unpushed_batches() >= self.GR.max_unpushed_batches)  # Condition C = Whether the batch pipeline is congested (condition M is already guaranteed from pushing batches above, so only need to check the second half of condition C)

			# Generate requests based on the available samples and task state (sets condition G if generation_done is True) and add them to the request pool (nullifies condition P)
			if batch_congestion:
				generation_done = False  # Condition G = Batch congestion is preventing further requests from being generated, so there may be more once the congestion is over
			else:
				generation_done = self.call_generate_requests()  # Condition G = Returns whether there are no more requests that can be generated right now

			# Commit all pooled requests to the request queue (may also happen intermittently inside generate_requests() call above, ensures condition P, nullifies conditions Q and B)
			# Create local batches out of the requests in the request queue, including a non-full trailing batch if condition G, otherwise some requests may remain in the queue (ensures condition B, ensures condition Q if G, nullifies conditions L, M and C, potentially nullifies condition V)
			# Push any pushable local batches to the server up to the extent of the push limits (ensures condition M, potentially sets condition L, nullifies condition R)
			batch_congestion = self.commit_requests(batch=True, force_batch=generation_done, push=True)  # Condition C = Returns whether the batch pipeline is congested

			# Check whether the step and/or entire task is completed (nothing more to do)
			assert self.GR.PQ.pool_len <= 0 and (self.GR.num_finished_batches() <= 0 or self.GR.dryrun)  # Assert PF (conditions B and M are less simple to assert, but should also always be true here)
			if (self.GR.num_unpushable_batches() <= 0 or direct_limits_reached) and ((generation_done and self.GR.PQ.queue_len <= 0) or batch_congestion):  # Condition E = PBVMF(GQ + C) = V(GQ + C) as conditions P, B, M and F are all guaranteed due to the commands above, by just reaching this line
				all_done = (self.GR.num_unpushed_batches() <= 0 and self.GR.num_unfinished_batches() <= 0)  # Condition D = ELR = There are no unpushed local batches and no unfinished remote batches (condition E must be met simply by reaching this line)
				break

		log.info('\u2500' * 120)
		log.info(f"Finished step {self.step_num}")
		return not all_done

	# Make and process a series of direct requests
	def direct_requests(self, reqs: Union[Iterable[Union[gpt_requester.GPTRequest, gpt_requester.GPTRequestItem, gpt_requester.CachedGPTRequest]], gpt_requester.GPTRequest, gpt_requester.GPTRequestItem, gpt_requester.CachedGPTRequest]) -> bool:
		# reqs = The requests to make and process using the direct API (raw, itemized and/or cached requests can be provided)
		# Returns a boolean whether direct limits were reached in the process (meaning that at least one request could not be fully completed due to them, or only-process mode)
		# This method performs direct API calls, updates the task/requester state, and adds the corresponding BatchState's to direct_history

		log.info('\u2500' * 120)
		direct_limits_reached = False
		for rstack, result, limited, _ in self.GR.direct_requests(reqs=reqs, yield_retry_reqs=False):
			if rstack is not None and result is not None:
				self.call_process_batch_result(result=result, rstack=rstack)
			if limited:
				direct_limits_reached = True

		return direct_limits_reached

	# Call the generate requests implementation
	def call_generate_requests(self) -> bool:
		# Passes on the return value of generate_requests()
		log.info('\u2500' * 120)
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
		# Assumption: The available samples, on the basis of which this method generates requests, are fixed for the entire duration that a task manager is entered, e.g. for an entire call to run().
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
						rstack.callback(revert_committed_state, meta=copy.deepcopy(self.T.committed_meta), samples_all=True, samples=copy.deepcopy(self.T.committed_samples))
					else:
						rstack.callback(revert_committed_state, meta=copy.deepcopy(self.T.committed_meta), samples_all=False, samples={sample_key: (copy.deepcopy(self.T.committed_samples[sample_key]) if sample_key in self.T.committed_samples else DELETE) for sample_key in sample_keys})

					for cached_req in cached_reqs:
						self.commit_cached_request(cached_req)

					self.validate_state(clean=False)
					self.task.save(rstack=rstack)

			log.info(f"Committed {len(cached_reqs)} generated requests")

		if batch and self.GR.Q:
			log.info(f"Attempting to {'force-' if force_batch else ''}create local batches from {len(self.GR.Q)} available committed requests...")
			num_unpushed_batches = self.GR.batch_requests(force=force_batch)
			if num_unpushed_batches > 0:
				log.info(f"The total number of unpushed local batches is now {num_unpushed_batches} ({self.GR.num_pushable_batches()} pushable and {self.GR.num_unpushable_batches()} unpushable)")

		if push:
			batch_congestion = self.push_batches()
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

	# Push as many pushable local batches as possible to the remote server
	def push_batches(self) -> bool:
		# Returns whether the batch pipeline is (now) congested
		if self.GR.num_pushable_batches() > 0:
			log.info('\u2500' * 120)
			log.info("Checking whether any local batches can be pushed to the remote server...")
			return self.GR.push_batches()  # Returns whether the batch pipeline is congested
		else:  # No pushable local batches => Condition M satisfied
			return self.GR.num_unpushed_batches() >= self.GR.max_unpushed_batches

	# Execute, process and clean up after as many unpushable local batches as possible (up to first reaching of direct limits)
	def process_unpushable_batches(self) -> tuple[int, bool]:
		# Returns how many unpushable local batches are still left after processing, and whether direct limits were reached in the process (meaning that at least one request could not be fully completed due to them, or only-process mode)
		# There will only ever be unpushable local batches left over if the direct limits were reached

		num_unpushable_batches = self.GR.num_unpushable_batches()
		direct_limits_reached = self.GR.only_process

		if num_unpushable_batches > 0:
			log.info('\u2500' * 120)
			log.info(f"Attempting to process the {num_unpushable_batches} available local unpushable batches...")
			for rstack, result, limited in self.GR.process_unpushable_batches():  # Either processes all unpushable batches one virtual (sub-)batch at a time, or eventually yields limited=True and discontinues, or raises an exception if an unresolvable issue occurs
				if rstack is not None and result is not None:
					self.call_process_batch_result(result=result, rstack=rstack)
				if limited:
					direct_limits_reached = True
			num_unpushable_batches = self.GR.num_unpushable_batches()

		assert num_unpushable_batches <= 0 or direct_limits_reached  # Condition V
		return num_unpushable_batches, direct_limits_reached

	# Process and clean up after any finished remote batches
	def process_batches(self) -> int:
		# Returns the current number of unfinished remote batches (after the remote batch status updates)
		# This method checks the remote for updated batch statuses, collects the results of any finished batches, updates the task/requester state, and cleans up that batch (also from the remote), moving the corresponding BatchState from batches to batch_history
		log.info('\u2500' * 120)
		for rstack, result in self.GR.process_batches():
			self.call_process_batch_result(result=result, rstack=rstack)
		assert self.GR.num_finished_batches() <= 0 or self.GR.dryrun  # Condition F
		return self.GR.num_unfinished_batches()

	# Call the process batch result implementation
	def call_process_batch_result(self, result: gpt_requester.BatchResult, rstack: utils.RevertStack) -> bool:
		# result => The result of a batch, to be processed and used to update the task output file and task state
		# rstack => A RevertStack so that the actions of this method are perfectly reversible in the case of an exception
		# Returns whether the task state (self.T) and output (self.D) were saved (either both or neither are saved)
		if self.process_batch_result(result=result, rstack=rstack):
			self.validate_state(clean=False)
			self.output.validate()
			self.task.save(rstack=rstack)
			self.output.save(rstack=rstack)
			return True
		else:
			return False

	# Process a batch result and accordingly update the task state and output files (must be a perfectly reversible operation managed by the RevertStack rstack)
	def process_batch_result(self, result: gpt_requester.BatchResult, rstack: utils.RevertStack) -> bool:
		# result => The result of a batch, to be processed and used to update the task output file and task state
		# rstack => A RevertStack so that the actions of this method are perfectly reversible in the case of an exception
		# Returns whether either the task state (self.T) or output (self.D) was modified, and thus that both need to be saved
		#
		# The final batch state---including the final remote batch status (result.batch.remote.batch: openai.type.Batch), API metrics (result.batch.metrics: APIMetrics), and true cost (result.batch.true_tokens_cost: TokensCost)---is available in result.batch (BatchState)
		# The remote batch completion duration is available as an integer number of seconds (result.duration) and as a formatted hours and minutes string (result.duration_hmin)
		# If the batch encountered any general/global errors then these are listed in result.errors (list[openai.types.BatchError])---however, there may nonetheless be valid responses even if there are such errors, so best to just immediately check the responses instead if that's all that counts
		# The main results/responses to process are the values of the result.info dict, which maps request IDs (int, returned from self.GR.add_request() / self.GR.add_requests()) to ResultInfo instances.
		# The following information is available for each ResultInfo instance 'info':
		#   - The input request payload (info.req_payload: dict[str, Any]) and metadata (info.req_info.meta: Optional[dict[str, Any]])
		#   - If a response was received for the request (info.resp_info is not None), the response in Python class/parsed format (info.resp_info.payload: openai.types.chat.ChatCompletion/ParsedChatCompletion or similar depending on auto-parse and endpoint)
		#   - If an error occurred with the request and/or response (info.err_info is not None), the error (info.err_info: ErrorInfo) => At least one of info.err_info and info.resp_info will always be non-None UNLESS dry run is active
		#   - If warnings occurred while processing the response, the warnings (info.warn_infos: list[ErrorInfo])---warnings occur for example if multiple completion choices are requested and some choices fail somehow while others don't
		#   - Whether the request will be retried (info.retry: bool) and whether the current result counts towards the retry number (info.retry_counts: bool / e.g. batch cancellation or expiry by default does not count)
		#   - The default (on entering this method) value of info.retry is never True (and info.retry_counts is always True) IF there is no error present (info.err_info is None)
		#   - The field info.retry can be MODIFIED in this method (calling self.GR.update_result_retry() after modifying info.err_info is suggested) to set whether the request will get retried (e.g. because of a task-specific parsing or value failure)
		#   - Theoretically, info.req_payload, info.req_info.meta, info.retry_counts and info.req_info.retry_num can also be MODIFIED to affect/tweak the retry, but is not recommended in general (e.g. care needs to be taken not to change anything in the payload that breaks auto-parsing)
		# Statistics like the request pass ratio, and the number of requests that were successful, warned, errored, cancelled, expired, etc, can be found in result.stats (ResultStats)
		raise NotImplementedError

	# Perform any required post-processing of the task output data (called only once the entire task completes)
	def postprocess_output_data(self):
		# Refer to the postprocess_output __init__ argument for additional information when this method is called
		pass

#
# Task manager helpers
#

# Opinion adviser class
class OpinionAdviser:

	def __init__(self, opinions_min: int, opinions_max: int, confidence: float):
		# opinions_min = Minimum number of successful opinions required
		# opinions_max = Maximum number of opinions allowed
		# confidence = Opinion-based classification confidence required, in the interval (0, 1)
		self.opinions_min = opinions_min
		self.opinions_max = opinions_max
		self.confidence = confidence
		self.validate()

	def validate(self):
		if not isinstance(self.opinions_min, int) or not isinstance(self.opinions_max, int) or self.opinions_min < 1 or self.opinions_max < 1 or self.opinions_max < self.opinions_min:
			raise ValueError(f"Invalid opinions specification: Min {self.opinions_min}, Max {self.opinions_max}")
		if not (isinstance(self.confidence, float) and 0.0 < self.confidence < 1.0):
			raise ValueError(f"Invalid confidence specification: {self.confidence}")

	def min_responses_reqd(self, num_failed: int, opinions: Iterable[Any]) -> int:
		# num_failed = Number of failed responses (responses that did not lead to an opinion)
		# opinions = The opinions resulting from all succeeded responses (responses that led to an opinion), where each individual opinion is non-None and compatible with collections.Counter (i.e. hashable type with appropriate equality check)
		# Returns the minimum number of total responses that could result in the done condition being satisfied (i.e. that there is a conclusion, e.g. due to confidence or opinions_max)
		# Aims to achieve num_succeeded >= opinions_min and most_common_count / num_succeeded >= confidence, without allowing opinions_max to be exceeded however (unless we already have more responses than that of course)
		# Failed requests are effectively ignored, aside from counting towards opinions_max and thereby implicitly taking away space for succeeded requests (of which we aim to have at least opinions_min)
		if not isinstance(opinions, collections.abc.Sequence):
			opinions = tuple(opinions)
		num_succeeded = len(opinions)
		_, _, most_common_count = self.resolve_opinions(opinions=opinions)
		min_responses = num_failed + max(self.opinions_min, math.ceil((num_succeeded - most_common_count) / (1 - self.confidence)))
		return max(num_failed + num_succeeded, min(self.opinions_max, min_responses))

	def get_conclusion(self, num_failed: int, opinions: Iterable[T]) -> tuple[bool, bool, Counter[T], Optional[T], int]:
		# num_failed = Number of failed responses (responses that did not lead to an opinion)
		# opinions = The opinions resulting from all succeeded responses (responses that led to an opinion), where each individual opinion is non-None and compatible with collections.Counter (i.e. hashable type with appropriate equality check)
		# Returns whether there is a conclusion, whether that conclusion is due to confidence (e.g. as opposed to due to opinions_max), a counter of the opinions, the most common opinion (None if there are no succeeded opinions), and the corresponding count
		if not isinstance(opinions, collections.abc.Sequence):
			opinions = tuple(opinions)
		num_succeeded = len(opinions)
		opinions_counter, most_common_opinion, most_common_count = self.resolve_opinions(opinions=opinions)
		assert opinions_counter.total() == num_succeeded
		is_confident = (num_succeeded >= self.opinions_min and most_common_count / num_succeeded >= self.confidence)
		have_conclusion = (is_confident or num_failed + num_succeeded >= self.opinions_max)
		return have_conclusion, is_confident, opinions_counter, most_common_opinion, most_common_count

	@classmethod
	def resolve_opinions(cls, opinions: Iterable[T]) -> tuple[Counter[T], Optional[T], int]:
		# opinions = The opinions to resolve, where each individual opinion is non-None and compatible with collections.Counter (i.e. hashable type with appropriate equality check)
		# Returns a counter of the opinions, the most common opinion (None if there are no succeeded opinions), and the corresponding count
		opinions_counter: Counter[T] = collections.Counter(opinions)
		most_common_list: list[tuple[T, int]] = opinions_counter.most_common(n=1)
		if most_common_list:
			return opinions_counter, *most_common_list[0]  # noqa / Returns: opinions_counter, most_common_opinion, most_common_count
		else:
			return opinions_counter, None, 0
# EOF
