# General utilities

# Imports
import os
import sys
import copy
import enum
import json
import math
import time
import types
import signal
import shutil
import inspect
import logging
import pathlib
import argparse
import tempfile
import importlib
import itertools
import contextlib
import collections
import dataclasses
import typing
from typing import TYPE_CHECKING, Any, Type, Self, Union, Optional, Iterable, TextIO, BinaryIO, ContextManager, Protocol, Callable, Counter, TypeVar
from types import FrameType
import filelock
import pydantic
import wandb

# Types
DataclassInstance = TypeVar('DataclassInstance')  # Generic dataclass instance
T = TypeVar('T')                                  # Generic type variable
C = TypeVar('C', bound=type)                      # Generic type variable for class types

# Logging configuration
logging.getLogger("filelock").setLevel(logging.WARNING)

# Constants
NONE = object()  # Sentinel object that can be used to determine whether a (possibly None) keyword argument was passed to a function

#
# Logging/Printing
#

# Custom color formatter for the logger
class ColorFormatter(logging.Formatter):

	FMT = "[%(levelname)s][%(asctime)s] %(message)s"
	DATEFMT = "%d-%b-%y %H:%M:%S"
	LEVEL_REMAP = {
		'DEBUG': '\x1b[38;21mDEBUG\x1b[0m',
		'INFO': '\x1b[38;5;39m INFO\x1b[0m',
		'WARNING': '\x1b[38;5;226m WARN\x1b[0m',
		'ERROR': '\x1b[38;5;196mERROR\x1b[0m',
		'CRITICAL': '\x1b[31;1mFATAL\x1b[0m',
	}

	def format(self, record: logging.LogRecord) -> str:
		record.levelname = self.LEVEL_REMAP.get(record.levelname, record.levelname)
		return super().format(record)

# Configure logging
def configure_logging() -> logging.Logger:
	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.set_name('console')
	stream_handler.setFormatter(ColorFormatter(fmt=ColorFormatter.FMT, datefmt=ColorFormatter.DATEFMT))
	logging.basicConfig(level=logging.INFO, format=ColorFormatter.FMT, handlers=[stream_handler])
	return logging.getLogger()

# In-place printing (replace the current line and don't advance to the next line)
def print_in_place(obj: Any):
	print(f"\x1b[2K\r{obj}", end='', flush=True)

# Clear current line
def print_clear_line():
	print("\x1b[2K\r", end='', flush=True)

# Format a duration as a single nearest appropriate integral time unit (e.g. one of 15s, 3m, 6h, 4d)
def format_duration(seconds: Union[int, float]) -> str:
	duration = abs(seconds)
	round_duration = int(duration)
	if round_duration < 60:
		round_duration_unit = 's'
	else:
		round_duration = int(duration / 60)
		if round_duration < 60:
			round_duration_unit = 'm'
		else:
			round_duration = int(duration / 3600)
			if round_duration < 24:
				round_duration_unit = 'h'
			else:
				round_duration = int(duration / 86400)
				round_duration_unit = 'd'
	return f"{'-' if seconds < 0 else ''}{round_duration}{round_duration_unit}"

# Format a duration as hours and minutes
def format_duration_hmin(seconds: Union[int, float]) -> str:
	duration = abs(seconds)
	hours = int(duration / 3600)
	minutes = int(duration / 60 - hours * 60)
	if hours == 0:
		return f"{'-' if seconds < 0 else ''}{minutes}m"
	else:
		return f"{'-' if seconds < 0 else ''}{hours}h{minutes}m"

# Format a size in bytes using the most appropriate IEC unit (e.g. 24B, 16.8KiB, 5.74MiB)
def format_size_iec(size: int, fmt: str = '.3g') -> str:
	if size < 0:
		raise ValueError(f"Size cannot be negative: {size}")
	base = 1
	thres = 1000
	for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB'):
		if size < thres:
			return f'{size / base:{fmt}}{unit}'
		base <<= 10
		thres <<= 10
	return f'{size / base:{fmt}}YiB'

# Format a size in bytes using the most appropriate SI unit (e.g. 24B, 16.8KB, 5.74MB)
def format_size_si(size: int, fmt: str = '.3g') -> str:
	if size < 0:
		raise ValueError(f"Size cannot be negative: {size}")
	base = 1
	thres = 1000
	for unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB'):
		if size < thres:
			return f'{size / base:{fmt}}{unit}'
		base *= 1000
		thres *= 1000
	return f'{size / base:{fmt}}YB'

# Get the full class spec of a class (can also provide an instance of that class)
def get_class_str(obj: Any) -> str:
	obj_type = type(obj)
	if obj_type is type:
		obj_type = obj
	if obj_type.__module__ == 'builtins':
		return obj_type.__qualname__
	else:
		return f'{obj_type.__module__}.{obj_type.__qualname__}'

# Get the full spec of a type or type annotation (e.g. supports types like dict[str, list[int]])
def get_type_str(typ: type) -> str:
	typ_str = format(typ)
	if typ_str.startswith("<class '") and typ_str.endswith("'>"):
		return typ_str[8:-2]
	else:
		return typ_str

# Mutable wandb run holder
if TYPE_CHECKING:
	class WandbRun(wandb.sdk.wandb_run.Run):
		pass
else:
	class WandbRun:

		class NoneCallable:
			def __call__(self, *args, **kwargs) -> None:
				return None
		NONE_CALLABLE = NoneCallable()

		run: Optional[wandb.sdk.wandb_run.Run]

		def __init__(self, run: Optional[wandb.sdk.wandb_run.Run] = None):
			self.set(run=run)

		def reset(self):
			self.run = None

		def set(self, run: Optional[wandb.sdk.wandb_run.Run]):
			self.run = run

		def __bool__(self) -> bool:
			return self.run is not None

		def __getattr__(self, name: str) -> Any:
			# Raise AttributeError in all cases unless the requested attribute is a known (non-property) method (and in that case just return None if the returned attribute value is called)
			if self.run is None:
				try:
					cls_attr = getattr(wandb.sdk.wandb_run.Run, name)
				except AttributeError:
					pass
				else:
					if callable(cls_attr) and inspect.isroutine(cls_attr):
						return self.NONE_CALLABLE
				raise AttributeError(f"Cannot get non-routine attribute '{name}' of {self.__class__.__name__} when internal run is None")
			else:
				return getattr(self.run, name)

#
# Error handling
#

# Delayed raise class
class DelayedRaise:

	def __init__(self):
		self.msgs: Counter[str] = collections.Counter()
		self.section_msgs: Counter[str] = collections.Counter()

	def new_section(self):
		self.section_msgs.clear()

	def add(self, msg: str, count: int = 1):
		# msg = Raise message to add
		# count = Multiplicity of the message (nothing is added if this is <=0)
		if count > 0:
			self.msgs[msg] += count
			self.section_msgs[msg] += count

	def have_errors(self) -> bool:
		# Returns whether there are any delayed errors
		return self.msgs.total() > 0

	def have_section_errors(self) -> bool:
		# Returns whether there are any delayed errors in the current section
		return self.section_msgs.total() > 0

	def raise_on_error(self, base_msg: str = 'Encountered errors'):
		# base_msg = Base message of the raised exception
		if self.have_errors():
			raise RuntimeError(f"{base_msg}:{''.join(f'\n  {count} \xD7 {msg}' for msg, count in sorted(self.msgs.items()) if count > 0)}")

# Log summarizer that only logs the first N messages and then provides a summary message at the end with how many further messages were omitted and/or how many there were in total
class LogSummarizer:

	def __init__(self, log_fn: Callable[[str], None], show_msgs: int):
		# log_fn = Logging function to use (e.g. log.error where log is a logging.Logger)
		# show_msgs = Number of first-received messages to log, before waiting until finalization to summarize how many further/total messages occured
		self.log_fn = log_fn
		self.show_msgs = show_msgs
		self.num_msgs = 0

	def reset(self):
		self.num_msgs = 0

	def log(self, msg: str) -> bool:
		# msg = Message to conditionally log
		# Returns whether the message was logged
		self.num_msgs += 1
		if self.num_msgs <= self.show_msgs:
			self.log_fn(msg)
			return True
		else:
			return False

	def finalize(self, msg_fn: Callable[[int, int], str]) -> bool:
		# msg_fn = Callable that generates a summary message string to log given how many messages were omitted and how many messages there were in total (e.g. lambda num_omitted, num_total: f"A further {num_omitted} messages occurred for a total of {num_total}")
		# Returns whether a final message was logged (a final message is only logged if more messages occurred than were shown)
		if self.num_msgs > self.show_msgs:
			self.log_fn(msg_fn(self.num_msgs - self.show_msgs, self.num_msgs))
			return True
		else:
			return False

#
# Dataclasses
#

# Convert a dataclass to JSON (supports only JSON-compatible types (or subclasses): dataclass, pydantic model, dict, list, tuple, str, float, int, bool, None)
def json_from_dataclass(obj: DataclassInstance, file: Optional[TextIO] = None, **kwargs) -> Optional[str]:
	obj_dict = dict_from_dataclass(obj=obj, json_mode=True)
	dumps_kwargs = dict(ensure_ascii=False, indent=2)
	dumps_kwargs.update(kwargs)
	if file is None:
		return json.dumps(obj_dict, **dumps_kwargs)
	else:
		json.dump(obj_dict, file, **dumps_kwargs)  # noqa
		return None

# Convert JSON to a dataclass (supports only JSON-compatible types (or subclasses): dataclass, pydantic model, dict, list, tuple, str, float, int, bool, None)
def dataclass_from_json(cls: Type[DataclassInstance], json_data: Union[TextIO, str]) -> DataclassInstance:
	data = json.loads(json_data) if isinstance(json_data, str) else json.load(json_data)
	return dataclass_from_dict(cls=cls, data=data, json_mode=True)

# Convert a dataclass to a dict (recurses into exactly only nested dataclass, dict, list, tuple, namedtuple (or subclasses) instances, just reuses atomic types like int/float (see dataclasses._ATOMIC_TYPES), dumps pydantic models to dict without explicit recursion, and uses copy.deepcopy() on everything else)
def dict_from_dataclass(obj: DataclassInstance, json_mode: bool = False) -> dict[str, Any]:
	if not dataclasses._is_dataclass_instance(obj):  # noqa
		raise TypeError(f"Object must be a dataclass instance but got an object of type: {type(obj)}")
	return _dict_from_dataclass_inner(obj=obj, json_mode=json_mode)

# Inner function for converting a dataclass to a dict (implementation adapted from dataclasses.asdict() @ Python 3.12.7)
def _dict_from_dataclass_inner(obj: Any, json_mode: bool) -> Any:
	if type(obj) in dataclasses._ATOMIC_TYPES:  # noqa
		return obj
	elif dataclasses._is_dataclass_instance(obj):  # noqa
		return {f.name: _dict_from_dataclass_inner(obj=getattr(obj, f.name), json_mode=json_mode) for f in dataclasses.fields(obj)}
	elif isinstance(obj, pydantic.BaseModel):
		return obj.model_dump(mode='json' if json_mode else 'python', warnings='error')
	elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
		return type(obj)(*[_dict_from_dataclass_inner(obj=v, json_mode=json_mode) for v in obj])
	elif isinstance(obj, (list, tuple)):
		return type(obj)(_dict_from_dataclass_inner(obj=v, json_mode=json_mode) for v in obj)
	elif isinstance(obj, dict):
		if isinstance(obj, collections.Counter):
			return type(obj)({_dict_from_dataclass_inner(obj=k, json_mode=json_mode): c for k, c in obj.items()})
		elif hasattr(type(obj), 'default_factory'):
			result = type(obj)(getattr(obj, 'default_factory'))
			for k, v in obj.items():
				result[_dict_from_dataclass_inner(obj=k, json_mode=json_mode)] = _dict_from_dataclass_inner(obj=v, json_mode=json_mode)
			return result
		return type(obj)((_dict_from_dataclass_inner(obj=k, json_mode=json_mode), _dict_from_dataclass_inner(obj=v, json_mode=json_mode)) for k, v in obj.items())
	else:
		return copy.deepcopy(obj)

# Convert a dict to a dataclass (recurses into exactly only nested dataclass, dict, list, tuple, namedtuple (or subclasses) instances, just reuses atomic types like int/float (see dataclasses._ATOMIC_TYPES), validates pydantic models from dict without explicit recursion, and uses copy.deepcopy() on everything else, can deal with simple cases of Union/Optional/Any/Ellipsis)
def dataclass_from_dict(cls: Type[DataclassInstance], data: dict[str, Any], json_mode: bool = False) -> DataclassInstance:
	if not dataclasses.is_dataclass(cls):
		raise TypeError(f"Class must be a dataclass type: {get_class_str(cls)}")
	return _dataclass_from_dict_inner(typ=cls, data=data, json_mode=json_mode)

# Inner function for converting nested data structures as appropriate to dataclasses
def _dataclass_from_dict_inner(typ: Any, data: Any, json_mode: bool) -> Any:
	generic_typ = typing.get_origin(typ) or typ  # e.g. dict[str, Any] -> dict
	if generic_typ is typing.Any:
		ret = copy.deepcopy(data)
	elif generic_typ is typing.Union:  # Also covers Optional...
		uniontyps = typing.get_args(typ)
		if not uniontyps:
			ret = copy.deepcopy(data)
		else:
			for uniontyp in uniontyps:
				generic_uniontyp = typing.get_origin(uniontyp) or uniontyp
				if isinstance(data, generic_uniontyp):
					ret = _dataclass_from_dict_inner(typ=uniontyp, data=data, json_mode=json_mode)
					break
			else:
				ret = _dataclass_from_dict_inner(typ=uniontyps[0], data=data, json_mode=json_mode)  # This is expected to usually internally raise an error as the types don't directly match...
	else:
		assert isinstance(generic_typ, type)
		if isinstance(data, int) and generic_typ is float:
			data = float(data)
		if json_mode:
			if isinstance(data, list) and generic_typ is not list and issubclass(generic_typ, (tuple, list)):
				if issubclass(generic_typ, tuple) and hasattr(generic_typ, '_fields'):
					data = generic_typ(*data)
				else:
					data = generic_typ(data)
			elif isinstance(data, dict) and issubclass(generic_typ, dict):
				key_typ = typing.get_args(typ)[0]
				generic_key_typ = typing.get_origin(key_typ) or key_typ
				if issubclass(generic_key_typ, (int, float)):
					data = {(generic_key_typ(key) if isinstance(key, str) else key): value for key, value in data.items()}
				if generic_typ is not dict:
					if issubclass(generic_typ, collections.Counter):
						data = generic_typ(data)
					elif hasattr(generic_typ, 'default_factory'):
						newdata = generic_typ(None)  # noqa / Note: We have no way of knowing what the default factory was prior to JSON serialization...
						for key, value in data.items():
							newdata[key] = value
						data = newdata
					else:
						data = generic_typ(data.items())
			elif isinstance(data, (int, float, complex, str, bytes, bool, types.NoneType)) and issubclass(generic_typ, enum.Enum):
				data = generic_typ(data)
		if dataclasses.is_dataclass(typ):
			if not (isinstance(data, dict) and all(isinstance(key, str) for key in data.keys())):
				raise TypeError(f"Invalid dict data for conversion to dataclass {get_class_str(typ)}: {data}")
			fields = dataclasses.fields(typ)
			field_names = set(field.name for field in fields)
			data_keys = set(data.keys())
			if field_names != data_keys:
				raise ValueError(f"Cannot construct {get_class_str(typ)} from dict that does not include exactly all the fields as keys for safety/correctness reasons => Dict is missing {sorted(field_names - data_keys)} and has {sorted(data_keys - field_names)} extra")
			field_types = typing.get_type_hints(typ)
			ret = typ(**{key: _dataclass_from_dict_inner(typ=field_types[key], data=value, json_mode=json_mode) for key, value in data.items()})
		elif issubclass(generic_typ, pydantic.BaseModel):
			if not (isinstance(data, dict) and all(isinstance(key, str) for key in data.keys())):
				raise TypeError(f"Invalid dict data for conversion to pydantic model {get_class_str(generic_typ)}: {data}")
			ret = generic_typ.model_validate(data, strict=None if json_mode else True)
		elif not isinstance(data, generic_typ):
			raise TypeError(f"Expected type {get_type_str(typ)} with generic type {get_class_str(generic_typ)} but got class {get_class_str(data)}: {data}")
		elif typ in dataclasses._ATOMIC_TYPES or (isinstance(typ, type) and issubclass(typ, enum.Enum)):  # noqa
			ret = data
		elif (is_tuple := issubclass(generic_typ, tuple)) and hasattr(generic_typ, '_fields'):
			if generic_typ.__annotations__:  # If defined using typing.NamedTuple...
				ret = generic_typ(*[_dataclass_from_dict_inner(typ=anntyp, data=value, json_mode=json_mode) for anntyp, value in zip(generic_typ.__annotations__.values(), data, strict=True)])  # Named tuples can't be directly constructed from an iterable
			else:
				ret = copy.deepcopy(data)
		else:
			subtyps = typing.get_args(typ)
			if not subtyps:
				ret = copy.deepcopy(data)
			elif is_tuple:
				if len(subtyps) == 2 and subtyps[-1] == Ellipsis:
					subtyps = (subtyps[0],) * len(data)
				ret = generic_typ(_dataclass_from_dict_inner(typ=subtyp, data=value, json_mode=json_mode) for subtyp, value in zip(subtyps, data, strict=True))
			elif issubclass(generic_typ, list):
				if len(subtyps) > 1:
					raise TypeError(f"Invalid multi-argument {get_class_str(generic_typ)} type annotation: {get_type_str(typ)}")
				ret = generic_typ(_dataclass_from_dict_inner(typ=subtyps[0], data=value, json_mode=json_mode) for value in data)
			elif issubclass(generic_typ, dict):
				if issubclass(generic_typ, collections.Counter):
					if len(subtyps) != 1:
						raise TypeError(f"Invalid {get_class_str(generic_typ)} type annotation: {get_type_str(typ)}")
					key_typ, = subtyps
					ret = generic_typ({_dataclass_from_dict_inner(typ=key_typ, data=key, json_mode=json_mode): count for key, count in data.items()})
				else:
					if len(subtyps) != 2:
						raise TypeError(f"Invalid {get_class_str(generic_typ)} type annotation: {get_type_str(typ)}")
					key_typ, value_typ = subtyps
					if hasattr(generic_typ, 'default_factory'):
						ret = generic_typ(getattr(data, 'default_factory'))
						for key, value in data.items():
							ret[_dataclass_from_dict_inner(typ=key_typ, data=key, json_mode=json_mode)] = _dataclass_from_dict_inner(typ=value_typ, data=value, json_mode=json_mode)
					else:
						ret = generic_typ((_dataclass_from_dict_inner(typ=key_typ, data=key, json_mode=json_mode), _dataclass_from_dict_inner(typ=value_typ, data=value, json_mode=json_mode)) for key, value in data.items())
			else:
				ret = copy.deepcopy(data)
		if not isinstance(ret, generic_typ):
			raise TypeError(f"Expected type {get_type_str(typ)} with generic type {get_class_str(generic_typ)} but got class {get_class_str(ret)}: {ret}")
	return ret

#
# OS/System
#

# Context manager that temporarily delays keyboard interrupts until the context manager exits
class DelayKeyboardInterrupt:

	def __init__(self):
		self.interrupted = False
		self.original_handler = None

	def __enter__(self) -> Self:
		self.interrupted = False
		self.original_handler = signal.signal(signal.SIGINT, self.sigint_handler)
		return self

	# noinspection PyUnusedLocal
	def sigint_handler(self, signum: int, frame: Optional[FrameType]):
		print("Received SIGINT: Waiting for next opportunity to raise KeyboardInterrupt... (use SIGTERM if this hangs)")
		self.interrupted = True

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		signal.signal(signal.SIGINT, self.original_handler)
		if self.interrupted:
			self.interrupted = False
			self.original_handler(signal.SIGINT, inspect.currentframe())
		self.original_handler = None
		return False

# Exit stack that allows convenient implementation of action reversion that takes effect only if an exception is encountered (based on contextlib.ExitStack @ Python 3.12.7)
# Note that whether the reversion callbacks are called (ALL of them) depends solely on whether the RevertStack receives an INITIAL exception, NOT whether an exception occurs during unwinding
class RevertStack(contextlib.ExitStack):

	def __init__(self):
		super().__init__()
		self._exit_callbacks_always = collections.deque()

	def pop_all(self) -> Self:
		new_stack = super().pop_all()
		new_stack._exit_callbacks_always = self._exit_callbacks_always
		self._exit_callbacks_always = collections.deque()
		return new_stack

	# noinspection PyShadowingBuiltins
	def push_always(self, exit: Callable) -> Callable:
		num_exit_callbacks = len(self._exit_callbacks)
		ret = self.push(exit=exit)
		assert len(self._exit_callbacks) == num_exit_callbacks + 1
		self._exit_callbacks_always.append(self._exit_callbacks[-1])
		return ret

	def enter_context_always(self, cm: ContextManager[T]) -> T:
		num_exit_callbacks = len(self._exit_callbacks)
		ret = self.enter_context(cm=cm)
		assert len(self._exit_callbacks) == num_exit_callbacks + 1
		self._exit_callbacks_always.append(self._exit_callbacks[-1])
		return ret

	def callback_always(self, callback: Callable, /, *args, **kwds) -> Callable:
		num_exit_callbacks = len(self._exit_callbacks)
		ret = self.callback(callback, *args, **kwds)
		assert len(self._exit_callbacks) == num_exit_callbacks + 1
		self._exit_callbacks_always.append(self._exit_callbacks[-1])
		return ret

	def __enter__(self) -> Self:
		return super().__enter__()

	def __exit__(self, *exc_details) -> bool:
		if exc_details[0] is None:
			self._exit_callbacks = self._exit_callbacks_always
			self._exit_callbacks_always = collections.deque()
		else:
			self._exit_callbacks_always.clear()
		# noinspection PyArgumentList
		return super().__exit__(*exc_details)

# Revert stack that can inherently collect data and log it to wandb on successful unwinding
# Note that ExitStack, RevertStack and LogRevertStack should be reusable in successive with statements, assuming they always either completely unwind or raise an exception that prevents subsequent with statements from executing
class LogRevertStack(RevertStack):

	def __init__(self, W: WandbRun):
		super().__init__()
		self.W = W
		self._log_data = {}

	def pop_all(self) -> Self:
		new_stack = super().pop_all()
		new_stack._log_data = self._log_data
		self._log_data = {}
		return new_stack

	def log(self, *args: dict[str, Any], flush: bool = False, **kwargs: Any):
		for arg in args:
			self._log_data.update(arg)
		if kwargs:
			self._log_data.update(kwargs)
		if flush and self._log_data:
			self.W.log(data=self._log_data)
			self._log_data = {}

	def __enter__(self) -> Self:
		return super().__enter__()

	def __exit__(self, *exc_details) -> bool:
		suppress_exc = super().__exit__(*exc_details)
		if self._log_data:
			if exc_details[0] is None:
				self.W.log(data=self._log_data)
			self._log_data = {}
		return suppress_exc

# Context manager that provides an ExitStack wrapped in DelayKeyboardInterrupt
@contextlib.contextmanager
def AtomicExitStack() -> ContextManager[contextlib.ExitStack[Optional[bool]]]:
	with DelayKeyboardInterrupt(), contextlib.ExitStack() as stack:
		yield stack

# Context manager that provides a RevertStack wrapped in DelayKeyboardInterrupt
@contextlib.contextmanager
def AtomicRevertStack() -> ContextManager[RevertStack]:
	with DelayKeyboardInterrupt(), RevertStack() as rstack:
		yield rstack

# Context manager that provides a LogRevertStack wrapped in DelayKeyboardInterrupt
@contextlib.contextmanager
def AtomicLogRevertStack(W: WandbRun) -> ContextManager[LogRevertStack]:
	with DelayKeyboardInterrupt(), LogRevertStack(W=W) as rstack:
		yield rstack

# Affix class
@dataclasses.dataclass
class Affix:                           # /path/to/file.ext --> /path/to/{prefix}file{root_suffix}.ext{suffix}
	prefix: Optional[str] = None       # /path/to/file.ext --> /path/to/{prefix}file.ext
	root_suffix: Optional[str] = None  # /path/to/file.ext --> /path/to/file{root_suffix}.ext
	suffix: Optional[str] = None       # /path/to/file.ext --> /path/to/file.ext{suffix}

# Context manager that performs a reversible write to a file by using a temporary file (in the same directory) as an intermediate and performing an atomic file replace (completely reversible on future exception if a RevertStack is supplied)
@contextlib.contextmanager
def SafeOpenForWrite(
	path: Union[str, pathlib.Path],        # path = Path of the file to safe-write to
	mode: str = 'w',                       # mode = File opening mode (should always be a 'write' or 'append' mode, i.e. including 'w' or 'a')
	*,                                     # Keyword arguments only beyond here
	temp_affix: Optional[Affix] = None,    # temp_affix = Affix to use for the temporary file path to write to before atomically replacing the target file with the temporary written file (defaults to a suffix of '.tmp', will also always have a random suffix added by the tempfile module to ensure a unique new file)
	rstack: Optional[RevertStack] = None,  # rstack = If a RevertStack is provided, callbacks are pushed to the stack to make all changes reversible on exception
	backup_affix: Optional[Affix] = None,  # backup_affix = If a RevertStack is provided, the affix to use for the backup file path (defaults to a suffix of '.bak', will also always have a random suffix added by the tempfile module to ensure a unique new file)
	**open_kwargs,                         # open_kwargs = Keyword arguments to provide to the internal call(s) to open()/tempfile.NamedTemporaryFile() (default kwargs of encoding='utf-8' and newline='\n' will be added unless the mode is binary or explicit alternative values are specified)
) -> ContextManager[Union[TextIO, BinaryIO]]:

	if isinstance(path, pathlib.Path):
		path = str(path)
	dirname, basename = os.path.split(path)
	root, ext = os.path.splitext(basename)

	write_mode = 'w' in mode
	append_mode = 'a' in mode
	binary_mode = 'b' in mode
	if write_mode and append_mode:
		raise ValueError(f"File opening mode cannot be both a write mode (w) and append mode (a): {mode}")
	elif not write_mode and not append_mode:
		raise ValueError(f"File opening mode must be a write mode (w) or an append mode (a): {mode}")
	if not binary_mode:
		open_kwargs.setdefault('encoding', 'utf-8')
		open_kwargs.setdefault('newline', '\n')
	mode_create = mode.replace('a', 'w') if append_mode else mode

	with contextlib.ExitStack() as stack:

		if temp_affix is None:
			temp_affix = Affix(prefix=None, root_suffix=None, suffix='.tmp')
		temp_base_name = f"{temp_affix.prefix or ''}{root}{temp_affix.root_suffix or ''}{ext}{temp_affix.suffix or ''}."

		if append_mode:

			with tempfile.NamedTemporaryFile(mode=mode_create, suffix=None, prefix=temp_base_name, dir=dirname, delete=False, **open_kwargs) as temp_file:
				@stack.callback
				def unlink_temp():
					with contextlib.suppress(FileNotFoundError):
						os.unlink(temp_file.name)

			try:
				shutil.copy2(src=path, dst=temp_file.name)
			except FileNotFoundError:
				pass

			with open(temp_file.name, mode=mode, **open_kwargs) as temp_file_append:
				yield temp_file_append

		else:

			with tempfile.NamedTemporaryFile(mode=mode, suffix=None, prefix=temp_base_name, dir=dirname, delete=False, **open_kwargs) as temp_file:
				@stack.callback
				def unlink_temp():
					with contextlib.suppress(FileNotFoundError):
						os.unlink(temp_file.name)
				yield temp_file

		if rstack is not None:

			if backup_affix is None:
				backup_affix = Affix(prefix=None, root_suffix=None, suffix='.bak')
			backup_base_name = f"{backup_affix.prefix or ''}{root}{backup_affix.root_suffix or ''}{ext}{backup_affix.suffix or ''}."

			with tempfile.NamedTemporaryFile(mode=mode_create, suffix=None, prefix=backup_base_name, dir=dirname, delete=False, **open_kwargs) as backup_file:
				@rstack.callback_always
				def unlink_backup():
					with contextlib.suppress(FileNotFoundError):
						os.unlink(backup_file.name)

			try:
				shutil.copy2(src=path, dst=backup_file.name)
				@rstack.callback  # noqa
				def revert_backup():
					os.replace(src=backup_file.name, dst=path)  # Internally atomic operation
			except FileNotFoundError:
				@rstack.callback
				def unlink_path():
					with contextlib.suppress(FileNotFoundError):
						os.unlink(path)

		os.replace(src=temp_file.name, dst=path)  # Internally atomic operation

# Safe unlink/delete a file (revertible operation using a RevertStack)
def safe_unlink(
	path: Union[str, pathlib.Path],  # path = Path of the file to unlink
	*,                               # Keyword arguments only beyond here
	rstack: RevertStack,             # rstack = RevertStack to use to ensure the unlinking is revertible
	affix: Optional[Affix] = None,   # affix = Affix to use for the backup file path used to ensure revertibility (defaults to a suffix of '.del', will also always have a random suffix added by the tempfile module to ensure a unique new file)
	missing_ok: bool = True,         # Whether an exception should be raised if the file to unlink does not exist (False) or whether such a case should be silently ignored (True)
) -> Optional[str]:                  # Returns the backup file path if a file existed and was revertibly unlinked, otherwise returns None

	if isinstance(path, pathlib.Path):
		path = str(path)
	dirname, basename = os.path.split(path)
	root, ext = os.path.splitext(basename)

	if affix is None:
		affix = Affix(prefix=None, root_suffix=None, suffix='.del')
	backup_base_name = f"{affix.prefix or ''}{root}{affix.root_suffix or ''}{ext}{affix.suffix or ''}."

	with tempfile.NamedTemporaryFile(mode='w', suffix=None, prefix=backup_base_name, dir=dirname, delete=False) as backup_file:
		@rstack.callback_always
		def unlink_backup():
			with contextlib.suppress(FileNotFoundError):
				os.unlink(backup_file.name)

	try:
		os.replace(src=path, dst=backup_file.name)
		@rstack.callback  # noqa
		def revert_backup():
			os.replace(src=backup_file.name, dst=path)  # Internally atomic operation
		return backup_file.name
	except FileNotFoundError:
		if not missing_ok:
			raise
		return None

# Reentrant lock file class that uses verbose prints to inform about the lock acquisition process in case it isn't quick
class LockFile:

	def __init__(
		self,
		path: str,                                # Lock file path (nominally *.lock extension)
		timeout: Optional[float] = None,          # Timeout in seconds when attempting to acquire the lock (<0 = No timeout, 0 = Instantaneous check, >0 = Timeout, Default = -1)
		poll_interval: Optional[float] = None,    # Polling interval when waiting for the lock (default 0.25s)
		status_interval: Optional[float] = None,  # Time interval to regularly print a status update when waiting for the lock (default 5s)
	):
		self.lock = filelock.FileLock(lock_file=path)
		self.timeout = timeout if timeout is not None else -1.0
		self.poll_interval = poll_interval if poll_interval is not None else 0.25
		self.status_interval = status_interval if status_interval is not None else 5.0
		if self.poll_interval < 0.01:
			raise ValueError(f"Poll interval must be at least 0.01s: {self.poll_interval}")
		if self.status_interval < 0.01:
			raise ValueError(f"Status interval must be at least 0.01s: {self.status_interval}")

	def __enter__(self) -> Self:

		start_time = time.perf_counter()
		print_time = start_time + self.status_interval
		print_duration_str = None

		if self.timeout >= 0:
			timeout_time = start_time + self.timeout
			timeout_str = f'/{format_duration(self.timeout)}'
		else:
			timeout_time = math.inf
			timeout_str = ''

		now = start_time
		while True:
			try:
				already_locked = self.lock.is_locked
				self.lock.acquire(timeout=max(min(timeout_time, print_time) - now, 0), poll_interval=self.poll_interval)
				if not already_locked:
					print_in_place(f"Successfully acquired lock in {format_duration(time.perf_counter() - start_time)}: {self.lock.lock_file}\n")
				return self
			except filelock.Timeout:
				now = time.perf_counter()
				if now >= timeout_time:
					print_in_place(f"Failed to acquire lock in {format_duration(now - start_time)}: {self.lock.lock_file}\n")
					raise
				elif now >= print_time:
					duration_str = format_duration(now - start_time)
					if duration_str != print_duration_str:
						print_in_place(f"Still waiting on lock after {duration_str}{timeout_str}: {self.lock.lock_file} ")
						print_duration_str = duration_str
					print_time += self.status_interval

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		self.lock.release()
		return False

# Get the size of an open file in bytes
def get_file_size(file: Union[TextIO, BinaryIO]) -> int:
	file.flush()  # Harmlessly does nothing if file is currently open for reading not writing
	return os.fstat(file.fileno()).st_size

#
# Types
#

# Protocol-based type annotation for configuration parameters represented as an object of attributes (e.g. argparse.Namespace, flat omegaconf.DictConfig)
class Config(Protocol):
	def __getattribute__(self, name: str) -> Any: ...

# Protocol-based type annotation for types that support vars()
class SupportsVars(Protocol):
	__dict__: dict[str, Any]

# Protocol-based type annotation for a typed sized iterable
class SizedIterable(Protocol[T]):
	def __iter__(self) -> Iterable[T]: ...
	def __len__(self) -> int: ...

# Serialized type cache
class SerialTypeCache:

	def __init__(self):
		self.type_map = {}

	def reset(self):
		self.type_map.clear()

	def cache_type(self, typ: type, verify: bool = False) -> str:
		# typ = The type to cache
		# verify = Whether to sanity check that type retrieval returns the exact original type
		# Returns the serialized string corresponding to the type

		assert '|' not in typ.__module__ and '|' not in typ.__qualname__
		serial = f'{typ.__module__}|{typ.__qualname__}'

		cached_typ = self.type_map.get(serial, None)
		if cached_typ is None:
			self.type_map[serial] = typ
		else:
			assert cached_typ is typ

		if verify:
			retrieved_typ = importlib.import_module(typ.__module__)
			for attr in typ.__qualname__.split('.'):
				retrieved_typ = getattr(retrieved_typ, attr)
			assert retrieved_typ is typ

		return serial

	def retrieve_type(self, serial: str) -> type:
		# serial = Serialized type string to retrieve the type for (must contain exactly one pipe separator)
		# Returns the retrieved/deserialized type

		typ = self.type_map.get(serial, None)

		if typ is None:
			typ_module, typ_qualname = serial.split('|')  # Errors if there is not exactly one pipe separator
			typ = importlib.import_module(typ_module)
			for attr in typ_qualname.split('.'):
				typ = getattr(typ, attr)
			self.type_map[serial] = typ

		return typ

# Ordered enumeration
class OrderedEnum(enum.Enum):

	def __ge__(self, other):
		if self.__class__ is other.__class__:
			return self.value >= other.value
		return NotImplemented

	def __gt__(self, other):
		if self.__class__ is other.__class__:
			return self.value > other.value
		return NotImplemented

	def __le__(self, other):
		if self.__class__ is other.__class__:
			return self.value <= other.value
		return NotImplemented

	def __lt__(self, other):
		if self.__class__ is other.__class__:
			return self.value < other.value
		return NotImplemented

# Enumeration with support for case-insensitive string lookup (case sensitive string lookup is already available by default)
# Note: If we have "class MyEnum(Enum): One = 1" then MyEnum(1) = MyEnum['One'] = MyEnum.One
class EnumLU(enum.Enum):

	@classmethod
	def from_str(cls, string, default=NONE):
		string = string.lower()
		for name, enumval in cls.__members__.items():
			if string == name.lower():
				return enumval
		if default is NONE:
			raise LookupError(f"Failed to convert case insensitive string to enum type {cls.__name__}: '{string}'")
		else:
			return default

	@classmethod
	def has_str(cls, string):
		string = string.lower()
		for name in cls.__members__:
			if string == name.lower():
				return True
		return False

# Ordered EnumLU enumeration
class OrderedEnumLU(OrderedEnum, EnumLU):
	pass

#
# Miscellaneous
#

# __init__ keyword argument class
@dataclasses.dataclass(frozen=True)
class InitKwarg:
	annotation: Any  # Type annotation of the __init__ keyword argument
	base_type: type  # Base type corresponding to the type annotation (takes the first option in the case of a Union)
	default: Any     # Default value of the __init__ keyword argument

# Decorator that adds __init__ method keyword argument introspection to a class (only considers non-self arguments that have a type annotation and default value)
def init_kwargs(cls: C) -> C:
	# cls = Class to decorate
	# Returns the updated class type, which now contains a __kwargs__ field (dictionary of all __init__ keyword arguments and their types/default values)
	init_type_hints = typing.get_type_hints(cls.__init__)  # Any __init__ arguments without type annotations simply do not appear in this dict
	cls.__kwargs__ = {}
	for name, param in inspect.signature(cls.__init__).parameters.items():
		if name != 'self' and name in init_type_hints and param.default != inspect.Parameter.empty:
			annotation = init_type_hints[name]
			base_type = typing.get_origin(annotation) or annotation
			if base_type is typing.Union:
				annotation_args = typing.get_args(annotation)
				assert annotation_args, "Union type annotation must contain at least one argument"
				arg_annotation = annotation_args[0]
				base_type = typing.get_origin(arg_annotation) or arg_annotation
			assert isinstance(base_type, type), f"Failed to infer __init__ argument type ({base_type}) from annotation: {annotation}"
			cls.__kwargs__[name] = InitKwarg(annotation=annotation, base_type=base_type, default=param.default)
	return cls

# If used @init_kwargs on cls: Populate a dictionary of __init__ keyword arguments based on an attribute-based config object (e.g. argparse.Namespace, flat omegaconf.DictConfig)
def get_init_kwargs(cls: C, cfg: Config, **kwargs) -> dict[str, Any]:
	# cls = Class type to get the __init__ keyword arguments for
	# cfg = Attribute-based config object to extract __init__ keyword arguments from (e.g. an instance of argparse.Namespace, or a flat omegaconf.DictConfig)
	# kwargs = Extra __init__ keyword arguments to set/override
	# Returns a dictionary of the extracted __init__ keyword arguments
	init_kwargs_ = {key: getattr(cfg, key) for key in cls.__kwargs__.keys() if hasattr(cfg, key)}
	init_kwargs_.update(kwargs)
	return init_kwargs_

# If used @init_kwargs on cls: Add an argument to an argparser or argparse group corresponding to an __init__ keyword argument (can use functools.partial to avoid passing cls, parser, defaults every time in multiple sequential calls)
def add_init_argparse(
	cls: C,                                                           # Class type to use as a reference for the __init__ keyword arguments (e.g. types and default values)
	parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup],  # noqa / Argument parser or group to add an argument to
	name: str,                                                        # Argument name (as per __init__ method signature)
	dest: Optional[str] = None,                                       # Override the argument destination (by default it is the same as the name)
	metavar: Union[str, tuple[str, ...], None] = None,                # Metavar to use for the argument
	unit: str = '',                                                   # String unit to use for the default value if one exists (if a space is needed between the default value and the unit then it must be part of this string)
	type: Optional[type] = None,                                      # noqa / Override the type specification of the argument (only if required and different to base type of type annotation)
	default: Any = NONE,                                              # Override the default value specification of the argument (if specified, overrides 'defaults' and cls.__kwargs__)
	defaults: Optional[dict[str, Any]] = None,                        # A dictionary of custom default values (overrides cls.__kwargs__ and can be overridden by 'default')
	help: str = '',                                                   # noqa / Argument help string
):
	if not help:
		raise ValueError(f"Help string must be provided for argument '{name}'")
	if dest is None:
		dest = name
	if default is NONE and defaults is not None:
		default = defaults.get(name, NONE)
	if type is None or default is NONE:
		if not hasattr(cls, '__kwargs__'):
			raise ValueError("Class must be decorated by @init_kwargs")
		elif name not in cls.__kwargs__:
			raise ValueError(f"Name '{name}' does not correspond to a type-annotated __init__ keyword argument of the class {get_class_str(cls)}")
		init_kwarg = cls.__kwargs__[name]  # Note: We only access __kwargs__ if it is actually necessary
		if type is None:
			type = init_kwarg.base_type  # noqa
		if default is NONE:
			default = init_kwarg.default
	if type is bool:
		if not (default is None or isinstance(default, bool)):
			raise ValueError(f"Default value for boolean argument '{name}' must be None or boolean: {get_class_str(type(default))}")
		if default:
			parser.add_argument(f'--no_{name}', dest=dest, action='store_false', default=default, help=help)
		else:
			parser.add_argument(f'--{name}', dest=dest, action='store_true', default=default, help=help)
	else:
		parser.add_argument(f'--{name}', dest=dest, type=type, default=default, metavar=metavar, help=help if default is None else f'{help} [default: {default}{unit}]')

# Check whether an iterable is in ascending order
def is_ascending(iterable: Iterable[Any], *, strict: bool) -> bool:
	if strict:
		return all(a < b for a, b in itertools.pairwise(iterable))
	else:
		return all(a <= b for a, b in itertools.pairwise(iterable))

# Check whether an iterable is in descending order
def is_descending(iterable: Iterable[Any], *, strict: bool) -> bool:
	if strict:
		return all(a > b for a, b in itertools.pairwise(iterable))
	else:
		return all(a >= b for a, b in itertools.pairwise(iterable))

# Resolve a default non-None value
def resolve(value: Any, default: Any) -> Any:
	return value if value is not None else default
# EOF
