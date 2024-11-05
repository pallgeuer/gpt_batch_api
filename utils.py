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
import itertools
import contextlib
import dataclasses
import typing
from typing import Any, Type, Self, Union, Optional, Iterable, TextIO, BinaryIO, ContextManager, Protocol
from types import FrameType
import filelock

# Types
DataclassInstance = typing.TypeVar('DataclassInstance')

# Logging configuration
logging.getLogger("filelock").setLevel(logging.WARNING)

#
# Logging/Printing
#

# In-place printing (replace the current line and don't advance to the next line)
def print_in_place(obj: Any):
	print(f"\x1b[2K\r{obj}", end='')

# Clear current line
def print_clear_line():
	print("\x1b[2K\r", end='')

# Format a duration as a single nearest appropriate integral time unit (e.g. one of 15s, 3m, 6h, 4d)
def format_duration(seconds: float) -> str:
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

# Format a size in bytes using the most appropriate unit (e.g. 24B, 16.8KiB, 5.74MiB)
def format_size(size: int, fmt: str = '.3g') -> str:
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
# Dataclasses
#

# Convert a dataclass to a dict (recurses into exactly only nested dataclass, dict, list, tuple, namedtuple (or subclasses) instances, just reuses atomic types like int/float (see dataclasses._ATOMIC_TYPES), and uses copy.deepcopy() on everything else)
def dict_from_dataclass(obj: DataclassInstance) -> dict[str, Any]:
	return dataclasses.asdict(obj)

# Convert a dict to a dataclass (recurses into exactly only nested dataclass, dict, list, tuple, namedtuple (or subclasses) instances, just reuses atomic types like int/float (see dataclasses._ATOMIC_TYPES), and uses copy.deepcopy() on everything else, can deal with simple cases of Union/Optional/Any/Ellipsis)
def dataclass_from_dict(cls: Type[DataclassInstance], data: dict[str, Any], json_mode: bool = False) -> DataclassInstance:
	if not dataclasses.is_dataclass(cls):
		raise TypeError(f"Class must be a dataclass type: {get_class_str(cls)}")
	return _dataclass_from_dict_inner(typ=cls, data=data, json_mode=json_mode)

# Convert a dataclass to JSON (supports only JSON-compatible types (or subclasses): dataclass, dict, list, tuple, str, float, int, bool, None)
def json_from_dataclass(obj: DataclassInstance, file: Optional[TextIO] = None, **kwargs) -> Optional[str]:
	dumps_kwargs = dict(ensure_ascii=False, indent=2)
	dumps_kwargs.update(kwargs)
	if file is None:
		return json.dumps(dict_from_dataclass(obj), **dumps_kwargs)
	else:
		json.dump(dict_from_dataclass(obj), file, **dumps_kwargs)  # noqa
		return None

# Convert JSON to a dataclass (supports only JSON-compatible types (or subclasses): dataclass, dict, list, tuple, str, float, int, bool, None)
def dataclass_from_json(cls: Type[DataclassInstance], json_data: Union[TextIO, str]) -> DataclassInstance:
	data = json.loads(json_data) if isinstance(json_data, str) else json.load(json_data)
	return dataclass_from_dict(cls=cls, data=data, json_mode=True)

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
			elif isinstance(data, dict) and generic_typ is not dict and issubclass(generic_typ, dict):
				if hasattr(generic_typ, 'default_factory'):
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

# Affix class
@dataclasses.dataclass
class Affix:                           # /path/to/file.ext --> /path/to/{prefix}file{root_suffix}.ext{suffix}
	prefix: Optional[str] = None       # /path/to/file.ext --> /path/to/{prefix}file.ext
	root_suffix: Optional[str] = None  # /path/to/file.ext --> /path/to/file{root_suffix}.ext
	suffix: Optional[str] = None       # /path/to/file.ext --> /path/to/file.ext{suffix}

# Context manager that performs an effectively atomic write to a file by using a temporary file (in the same directory) as an intermediate and performing an atomic file replace (completely reversible on future error if an ExitStack is supplied)
@contextlib.contextmanager
def SafeOpenForWrite(
	path: str,                                                     # path = Path of the file to safe-write to
	mode: str = 'w',                                               # mode = File opening mode (should always be a 'write' mode that ensures the file is created, i.e. including 'w' or 'x')
	*,                                                             # Keyword arguments only beyond here
	temp_affix: Optional[Affix] = None,                            # temp_affix = Affix to use for the temporary file path to write to before atomically replacing the target file with the temporary written file (defaults to a suffix of '.tmp')
	stack: Optional[contextlib.ExitStack[Optional[bool]]] = None,  # stack = If an ExitStack is provided, callbacks are pushed to the stack to revert all changes to the file if the stack unwinds with an exception
	backup_affix: Optional[Affix] = None,                          # backup_affix = If an ExitStack is provided, the affix to use for the backup file path (defaults to a suffix of '.bak')
	**open_kwargs,                                                 # open_kwargs = Keyword arguments to provide to the internal call to open() (default kwargs of encoding='utf-8' and newline='\n' will be added unless the mode is binary or explicit alternative values are specified)
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

	# noinspection PyUnusedLocal
	def unlink_temp_if_exc(exc_type, exc_val, exc_tb) -> bool:
		if exc_type is not None:
			with contextlib.suppress(FileNotFoundError):
				os.unlink(temp_path)
		return False

	with contextlib.ExitStack() as stack_:
		stack_.push(unlink_temp_if_exc)
		yield stack_.enter_context(open(temp_path, mode=mode, **open_kwargs))

	if stack is not None:

		if backup_affix is None:
			backup_affix = Affix(prefix=None, root_suffix=None, suffix='.bak')
		backup_path = os.path.join(dirname, f"{backup_affix.prefix or ''}{root}{backup_affix.root_suffix or ''}{ext}{backup_affix.suffix or ''}")

		def unlink_backup():
			with contextlib.suppress(FileNotFoundError):
				os.unlink(backup_path)
		stack.callback(unlink_backup)

		# noinspection PyUnusedLocal
		def revert_backup_if_exc(exc_type, exc_val, exc_tb) -> bool:
			if exc_type is not None:
				os.replace(src=backup_path, dst=path)
			return False

		try:
			shutil.copy2(src=path, dst=backup_path)
			stack.push(revert_backup_if_exc)
		except FileNotFoundError:
			pass

	try:
		os.replace(src=temp_path, dst=path)  # Internally atomic operation
	except OSError:
		with contextlib.suppress(FileNotFoundError):
			os.unlink(temp_path)
		raise

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

# Context manager that extends DelayKeyboardInterrupt to also by default provide an ExitStack that can be used to unwind partial operations in case one of the operations raises an exception
@contextlib.contextmanager
def AtomicExitStack() -> ContextManager[contextlib.ExitStack[Optional[bool]]]:
	with DelayKeyboardInterrupt(), contextlib.ExitStack() as stack:
		yield stack

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
				print_clear_line()
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

# Protocol-based type annotation for configuration parameters represented as an object of attributes (e.g. argparse.Namespace, flat omegaconf.DictConfig)
class Config(Protocol):

	def __getattribute__(self, name: str) -> Any:
		...
# EOF
