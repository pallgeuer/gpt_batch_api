# General utilities

# Imports
import copy
import json
import signal
import inspect
import dataclasses
import typing
from typing import Any, Type

# Types
DataclassInstance = typing.TypeVar('DataclassInstance')

#
# Logging/printing
#

# In-place printing (replace the current line and don't advance to the next line)
def print_in_place(obj: Any):
	print(f"\x1b[2K\r{obj}", end='')

# Clear current line
def print_clear_line():
	print("\x1b[2K\r", end='')

# Format a duration as a single nearest appropriate integral time unit (e.g. one of 15s, 3m, 6h, 4d)
def format_duration_unit(seconds: float) -> str:
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
def json_from_dataclass(obj: DataclassInstance, **kwargs) -> str:
	dumps_kwargs = dict(ensure_ascii=False, indent=2)
	dumps_kwargs.update(kwargs)
	return json.dumps(dict_from_dataclass(obj), **dumps_kwargs)

# Convert JSON to a dataclass (supports only JSON-compatible types (or subclasses): dataclass, dict, list, tuple, str, float, int, bool, None)
def dataclass_from_json(cls: Type[DataclassInstance], json_data: str) -> DataclassInstance:
	return dataclass_from_dict(cls=cls, data=json.loads(json_data), json_mode=True)

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
					newdata = generic_typ(None)  # noqa: We have no way of knowing what the default factory was prior to JSON serialization...
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
# Signals
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
	def sigint_handler(self, signum, frame):
		print("Received SIGINT: Waiting for next opportunity to raise KeyboardInterrupt... (use SIGTERM if this hangs)")
		self.interrupted = True

	def __exit__(self, exc_type, exc_val, exc_tb):
		signal.signal(signal.SIGINT, self.original_handler)
		if self.interrupted:
			self.interrupted = False
			self.original_handler(signal.SIGINT, inspect.currentframe())
		self.original_handler = None
# EOF
