# General utilities

# Imports
import os
import copy
import json
import math
import time
import signal
import shutil
import inspect
import logging
import importlib
import itertools
import contextlib
import collections
import dataclasses
import typing
from typing import Any, Type, Self, Union, Optional, Iterable, TextIO, BinaryIO, ContextManager, Protocol, Callable, Counter
from types import FrameType
import filelock
import pydantic

# Types
DataclassInstance = typing.TypeVar('DataclassInstance')
T = typing.TypeVar('T')

# Logging configuration
logging.getLogger("filelock").setLevel(logging.WARNING)

#
# Logging/Printing
#

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
			raise RuntimeError(f"{base_msg}:{''.join(f'\n{count} \xd7 {msg}' for msg, count in sorted(self.msgs.items()) if count > 0)}")

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
			ret = generic_typ.model_validate(data, strict=True)
		elif not isinstance(data, generic_typ):
			raise TypeError(f"Expected type {get_type_str(typ)} with generic type {get_class_str(generic_typ)} but got class {get_class_str(data)}: {data}")
		elif typ in dataclasses._ATOMIC_TYPES:  # noqa
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

	def __enter__(self):
		self.interrupted = False
		self.original_handler = signal.signal(signal.SIGINT, self.sigint_handler)

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
		else:
			self._exit_callbacks_always.clear()
		# noinspection PyArgumentList
		return super().__exit__(*exc_details)

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

# Affix class
@dataclasses.dataclass
class Affix:                           # /path/to/file.ext --> /path/to/{prefix}file{root_suffix}.ext{suffix}
	prefix: Optional[str] = None       # /path/to/file.ext --> /path/to/{prefix}file.ext
	root_suffix: Optional[str] = None  # /path/to/file.ext --> /path/to/file{root_suffix}.ext
	suffix: Optional[str] = None       # /path/to/file.ext --> /path/to/file.ext{suffix}

# Context manager that performs a reversible write to a file by using a temporary file (in the same directory) as an intermediate and performing an atomic file replace (completely reversible on future exception if a RevertStack is supplied)
@contextlib.contextmanager
def SafeOpenForWrite(
	path: str,                             # path = Path of the file to safe-write to
	mode: str = 'w',                       # mode = File opening mode (should always be a 'write' mode that ensures the file is created, i.e. including 'w' or 'x')
	*,                                     # Keyword arguments only beyond here
	temp_affix: Optional[Affix] = None,    # temp_affix = Affix to use for the temporary file path to write to before atomically replacing the target file with the temporary written file (defaults to a suffix of '.tmp')
	rstack: Optional[RevertStack] = None,  # rstack = If a RevertStack is provided, callbacks are pushed to the stack to make all changes reversible on exception
	backup_affix: Optional[Affix] = None,  # backup_affix = If a RevertStack is provided, the affix to use for the backup file path (defaults to a suffix of '.bak')
	**open_kwargs,                         # open_kwargs = Keyword arguments to provide to the internal call to open() (default kwargs of encoding='utf-8' and newline='\n' will be added unless the mode is binary or explicit alternative values are specified)
) -> ContextManager[Union[TextIO, BinaryIO]]:

	if 'w' not in mode and 'x' not in mode:
		raise ValueError(f"File opening mode must be a truncating write mode (w or x): {mode}")
	if 'b' not in mode:
		open_kwargs.setdefault('encoding', 'utf-8')
		open_kwargs.setdefault('newline', '\n')

	dirname, basename = os.path.split(path)
	root, ext = os.path.splitext(basename)

	if temp_affix is None:
		temp_affix = Affix(prefix=None, root_suffix=None, suffix='.tmp')
	temp_path = os.path.join(dirname, f"{temp_affix.prefix or ''}{root}{temp_affix.root_suffix or ''}{ext}{temp_affix.suffix or ''}")

	with RevertStack() as temp_rstack:
		@temp_rstack.callback
		def unlink_temp():
			with contextlib.suppress(FileNotFoundError):
				os.unlink(temp_path)
		yield temp_rstack.enter_context_always(open(temp_path, mode=mode, **open_kwargs))

	if rstack is not None:

		if backup_affix is None:
			backup_affix = Affix(prefix=None, root_suffix=None, suffix='.bak')
		backup_path = os.path.join(dirname, f"{backup_affix.prefix or ''}{root}{backup_affix.root_suffix or ''}{ext}{backup_affix.suffix or ''}")

		@rstack.callback_always
		def unlink_backup():
			with contextlib.suppress(FileNotFoundError):
				os.unlink(backup_path)

		try:
			shutil.copy2(src=path, dst=backup_path)
			@rstack.callback  # noqa
			def revert_backup():
				os.replace(src=backup_path, dst=path)  # Internally atomic operation
		except FileNotFoundError:
			@rstack.callback
			def unlink_path():
				with contextlib.suppress(FileNotFoundError):
					os.unlink(path)

	try:
		os.replace(src=temp_path, dst=path)  # Internally atomic operation
	except OSError:
		with contextlib.suppress(FileNotFoundError):
			os.unlink(temp_path)
		raise

# Lock file class that uses verbose prints to inform about the lock acquisition process in case it isn't quick
class LockFile:

	def __init__(
		self,
		path: str,                                # Lock file path (nominally *.lock extension)
		timeout: Optional[float] = None,          # Timeout in seconds when attempting to acquire the lock (<0 = No timeout, 0 = Instantaneous check, >0 = Timeout, Default = -1)
		poll_interval: Optional[float] = None,    # Polling interval when waiting for the lock (default 0.25s)
		status_interval: Optional[float] = None,  # Time interval to regularly print a status update when waiting for the lock (default 5s)
	):
		self.lock = filelock.FileLock(lock_file=path)
		self.timeout = timeout if timeout is not None else -1
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
				self.lock.acquire(timeout=max(min(timeout_time, print_time) - now, 0), poll_interval=self.poll_interval)
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

	def __getattribute__(self, name: str) -> Any:
		...

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

#
# Miscellaneous
#

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
# EOF
