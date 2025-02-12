# Utilities for counting and managing tokens

# Imports
import io
import re
import base64
import string
import binascii
import collections
import dataclasses
from typing import Any, Optional, Union, Sequence
import PIL.Image
import requests
import tiktoken
from .logger import log

# Input tokens count class
@dataclasses.dataclass(frozen=True)
class InputTokensCount:
	fixed: int          # Number of input tokens that are always required for every request
	msg_system: int     # Number of input tokens for system messages
	msg_user: int       # Number of input tokens for user messages
	msg_assistant: int  # Number of input tokens for assistant messages
	msg_tool: int       # Number of input tokens for tool messages
	msg_function: int   # Number of input tokens for function messages
	msg_other: int      # Number of input tokens for other messages
	msg_total: int      # Total number of input tokens for messages of all types
	meta: int           # Number of input tokens representing message metadata (non-extra non-content data)
	text: int           # Number of input tokens representing text content
	image: int          # Number of input tokens representing image content
	format: int         # Number of input tokens for specifying response format
	tools: int          # Number of input tokens for specifying available tools
	functions: int      # Number of input tokens for specifying available functions
	extra: int          # Number of hidden extra input tokens (including fixed)
	total: int          # Total number of input tokens

# Chat completions core tokens configuration
@dataclasses.dataclass
class CCCoreTokensConfig:
	tokens_per_response: int       # Fixed number of extra tokens required once in each response
	tokens_per_message: int        # Number of extra tokens required for each message
	tokens_per_message_gap: int    # Number of extra tokens required as separators between messages
	tokens_per_name: int           # Number of extra tokens required for every message that has a name
	tokens_last_text_unended: int  # Number of extra tokens required if the last content in a message is text that does not end in whitespace or punctuation

# Chat completions image tokens configuration
@dataclasses.dataclass
class CCImageTokensConfig:
	low_tokens: int        # Low detail: Fixed number of tokens required for every image
	high_tokens_base: int  # High detail: Number of base tokens required for every image
	high_tokens_tile: int  # High detail: Extra number of tokens required for every tile
	high_res: int          # High detail: The smaller dimension is downscaled (if it is larger) to this resolution prior to tiling
	high_tile: int         # High detail: Square tile size

# Chat completions function tokens configuration
@dataclasses.dataclass
class CCFuncTokensConfig:
	call_none: int           # Number of extra tokens required if the call mode is 'none'
	desc_strip_dot: int      # Maximum number of trailing dots to strip from description fields after whitespace has been stripped from both beginning and end
	func_init: int           # Number of extra tokens required to initialize each function
	func_desc_pre: int       # Number of extra tokens required for descriptions that are non-empty prior to stripping whitespace
	func_desc_post: int      # Number of extra tokens required for descriptions that are non-empty after stripping whitespace
	func_end: int            # Number of extra tokens required at the end of all functions
	root_props_init: int     # Number of fixed extra tokens required if the root properties are non-empty
	prop_key_init: int       # Number of extra tokens required for each non-ignored property key
	prop_enum_gap: int       # Number of extra tokens required for each enum item beyond the first
	prop_special_obj: int    # Number of extra tokens required for each object that either has no properties or is last in an object
	prop_special_last: int   # Number of extra tokens required if an array/enum is last in an object
	prop_desc_value: int     # Number of extra tokens required for descriptions of value properties
	prop_desc_obj: int       # Number of extra tokens required for descriptions of object properties
	prop_desc_oth: int       # Number of extra tokens required for descriptions of other properties (array/enum)
	prop_desc_item_obj: int  # Number of extra tokens required for descriptions of non-empty object array items
	prop_desc_item_oth: int  # Number of extra tokens required for descriptions of other array items
	prop_title_init: int     # Number of extra tokens required for titles

# Token estimator class
class TokenEstimator:

	assumed_completion_ratio: Optional[float]

	def __init__(self, assumed_completion_ratio: Optional[float], warn: str = 'once'):
		self.set_assumed_completion_ratio(assumed_completion_ratio)
		self.warn = warn
		if self.warn not in ('never', 'once', 'always'):
			raise ValueError(f"Invalid warn mode: {self.warn}")
		self.seen_warnings = set()

	def set_assumed_completion_ratio(self, assumed_completion_ratio: Optional[float]):
		self.assumed_completion_ratio = assumed_completion_ratio
		if self.assumed_completion_ratio is not None and not 0 <= self.assumed_completion_ratio <= 1:
			raise ValueError(f"Invalid assumed completion ratio: {self.assumed_completion_ratio}")

	def reset(self):
		self.seen_warnings.clear()

	def warning(self, msg: str):
		if self.warn == 'always':
			log.warning(msg)
		elif self.warn == 'once' and msg not in self.seen_warnings:
			log.warning(f"{msg} (WARN ONCE)")
			self.seen_warnings.add(msg)

	# Parse from a JSON payload all recognized content that counts towards input tokens and thereby estimate the input token requirements
	# Helpful source (accessed 28/10/2024): https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
	def payload_input_tokens(self, payload: dict[str, Any], endpoint: str) -> InputTokensCount:

		msg_tokens = collections.defaultdict(int)
		type_tokens = collections.defaultdict(int)
		meta = extra = 0

		if endpoint == '/v1/chat/completions':  # Chat completions endpoint

			model: str = payload['model']
			try:
				encoding = tiktoken.encoding_for_model(model)
			except KeyError:
				encoding = tiktoken.get_encoding('o200k_base')
				self.warning(f"Assuming '{encoding.name}' encoding for unrecognized model: {model}")

			core_cfg = CCCoreTokensConfig(
				tokens_per_response=3,
				tokens_per_message=3,
				tokens_per_message_gap=0,
				tokens_per_name=1,
				tokens_last_text_unended=0,
			)
			if re.match(r'(ft:)?(?:gpt-3.5|gpt-4|gpt-4-turbo|(?:chat)?gpt-4o(?:-mini)?)', model):
				pass
			elif re.match(r'(ft:)?o1', model):
				core_cfg.tokens_per_message_gap = 7
				core_cfg.tokens_per_name = 4
				core_cfg.tokens_last_text_unended = 1
			else:
				self.warning(f"Assuming default chat completions core tokens configuration for unrecognized model: {model}")
			special_end_tokens = (encoding.encode(' ')[0], encoding.encode('.')[0])

			image_cfg = CCImageTokensConfig(
				low_tokens=85,
				high_tokens_base=85,
				high_tokens_tile=170,
				high_res=768,
				high_tile=512,
			)
			supports_image = True
			if re.match(r'(ft:)?gpt-4o-mini', model):
				image_cfg.low_tokens = 2833
				image_cfg.high_tokens_base = 2833
				image_cfg.high_tokens_tile = 5667
			elif re.match(r'(ft:)?(?:gpt-4-turbo|(?:chat)?gpt-4o)', model):
				pass
			else:
				supports_image = False

			func_cfg = CCFuncTokensConfig(
				call_none=1,
				desc_strip_dot=4,
				func_init=6,
				func_desc_pre=1,
				func_desc_post=0,
				func_end=12,
				root_props_init=2,
				prop_key_init=2,
				prop_enum_gap=3,
				prop_special_obj=1,
				prop_special_last=1,
				prop_desc_value=2,
				prop_desc_obj=1,
				prop_desc_oth=1,
				prop_desc_item_obj=4,
				prop_desc_item_oth=5,
				prop_title_init=3,
			)
			supports_tools = True
			supports_json_schema = False
			if re.match(r'(ft:)?(?:chat)?gpt-4o(?:-mini)?', model):
				supports_json_schema = True
			elif re.match(r'(ft:)?(?:gpt-3.5-turbo|gpt-4|gpt-4-turbo)', model):
				func_cfg.desc_strip_dot = 3
				func_cfg.prop_desc_value = 3
				func_cfg.prop_desc_obj = 2
				func_cfg.prop_desc_oth = 2
				if re.match(r'(ft:)?gpt-4(?!-turbo)', model):
					func_cfg.func_desc_pre = 0
					func_cfg.func_desc_post = 2
				else:
					func_cfg.func_desc_post = 1
			else:
				supports_tools = False

			fixed = core_cfg.tokens_per_response
			extra += core_cfg.tokens_per_response
			for m, message in enumerate(payload['messages'], 1):
				role = message['role']
				message_tokens = core_cfg.tokens_per_message
				if m > 1:
					message_tokens += core_cfg.tokens_per_message_gap
				msg_tokens[role] += message_tokens
				extra += message_tokens
				for key, value in message.items():
					if key == 'content':
						if isinstance(value, str):
							value_tokens = len(encoding.encode(value))
							if value and value[-1] not in string.whitespace and value[-1] not in string.punctuation:
								value_tokens += core_cfg.tokens_last_text_unended
							msg_tokens[role] += value_tokens
							type_tokens['text'] += value_tokens
						elif value is not None:
							last_content_type = None
							last_content_tokens = None
							content_type = content_text = None
							for content_item in value:
								content_type = content_item['type']
								if content_type == 'text':
									content_text = content_item['text']
									content_tokens = encoding.encode(content_text)
									text_tokens = len(content_tokens)
									if last_content_type == 'text' and last_content_tokens and last_content_tokens[-1] in special_end_tokens:
										text_tokens -= 1
									last_content_tokens = content_tokens
									msg_tokens[role] += text_tokens
									type_tokens['text'] += text_tokens
								elif content_type == 'image_url':
									if not supports_image:
										self.warning(f"Assuming default chat completions image tokens configuration for unrecognized image model: {model}")
									image_spec = content_item['image_url']
									image_tokens = self.cc_image_input_tokens(
										url=image_spec['url'],
										detail=image_spec.get('detail', 'auto'),
										image_cfg=image_cfg,
									)
									msg_tokens[role] += image_tokens
									type_tokens['image'] += image_tokens
								else:
									self.warning(f"Ignoring input tokens corresponding to unrecognized content type: {content_type}")
								last_content_type = content_type
							if content_type == 'text' and content_text and content_text[-1] not in string.whitespace and content_text[-1] not in string.punctuation:
								msg_tokens[role] += core_cfg.tokens_last_text_unended
								type_tokens['text'] += core_cfg.tokens_last_text_unended
					elif isinstance(value, str):
						if key == 'name':
							msg_tokens[role] += core_cfg.tokens_per_name
							extra += core_cfg.tokens_per_name
						value_tokens = len(encoding.encode(value))
						msg_tokens[role] += value_tokens
						meta += value_tokens

			payload_response_format: Optional[dict[str, Any]] = payload.get('response_format', None)
			if payload_response_format:
				type_tokens['format'] += self.cc_response_format_input_tokens(
					encoding=encoding,
					response_format=payload_response_format,
					supports_json_schema=supports_json_schema,
					model=model,
					func_cfg=func_cfg,
				)

			payload_tools: Optional[Sequence[dict[str, Any]]] = payload.get('tools', None)
			if payload_tools:
				if not supports_tools:
					self.warning(f"Assuming default chat completions function tokens configuration for unrecognized tool model: {model}")
				type_tokens['tools'] += self.cc_tools_input_tokens(
					encoding=encoding,
					tools=payload_tools,
					tool_choice=payload.get('tool_choice', None),
					func_cfg=func_cfg,
				)

			payload_functions: Optional[Sequence[dict[str, Any]]] = payload.get('functions', None)
			if payload_functions:
				if not supports_tools:
					self.warning(f"Assuming default chat completions function tokens configuration for unrecognized function model: {model}")
				type_tokens['functions'] += self.cc_functions_input_tokens(
					encoding=encoding,
					functions=payload_functions,
					function_call=payload.get('function_call', None),
					func_cfg=func_cfg,
				)

		else:
			raise ValueError(f"Cannot estimate input tokens for unrecognized endpoint: {endpoint}")

		msg_system = msg_tokens['system']
		msg_user = msg_tokens['user']
		msg_assistant = msg_tokens['assistant']
		msg_tool = msg_tokens['tool']
		msg_function = msg_tokens['function']
		msg_total = sum(msg_tokens.values())
		msg_other = msg_total - (msg_system + msg_user + msg_assistant + msg_tool + msg_function)
		assert msg_other >= 0

		text = type_tokens['text']
		image = type_tokens['image']
		rformat = type_tokens['format']
		tools = type_tokens['tools']
		functions = type_tokens['functions']

		total = fixed + msg_total + rformat + tools + functions
		assert meta + text + image + rformat + tools + functions + extra == total

		return InputTokensCount(
			fixed=fixed,
			msg_system=msg_system,
			msg_user=msg_user,
			msg_assistant=msg_assistant,
			msg_tool=msg_tool,
			msg_function=msg_function,
			msg_other=msg_other,
			msg_total=msg_total,
			meta=meta,
			text=text,
			image=image,
			format=rformat,
			tools=tools,
			functions=functions,
			extra=extra,
			total=total,
		)

	# Chat completions: Estimate the number of input tokens required for an image
	# Helpful source (accessed 28/10/2024): https://platform.openai.com/docs/guides/vision/calculating-costs
	def cc_image_input_tokens(self, url: Union[str, tuple[int, int], None], detail: str, image_cfg: CCImageTokensConfig) -> int:

		if detail == 'low':
			return image_cfg.low_tokens

		image_res = self.cc_image_resolution(url) if isinstance(url, str) else url
		if image_res is None:
			return 0  # Ignore image if failed to identify its resolution
		width, height = image_res
		if width <= 0 or height <= 0:
			self.warning("Ignoring input tokens corresponding to image with non-positive resolution")
			return 0  # Ignore image if identified resolution has non-positive entries

		if min(width, height) > image_cfg.high_res:
			scale = image_cfg.high_res / min(width, height)
			width = round(width * scale)
			height = round(height * scale)

		num_tiles = ((width - 1) // image_cfg.high_tile + 1) * ((height - 1) // image_cfg.high_tile + 1)
		if detail == 'auto':
			self.warning("Cannot accurately determine number of input image tokens if detail level is 'auto' => Explicitly specify high/low if possible")
			if num_tiles <= 1:  # Assume low detail if resolution fits in a single tile (it is unclear how OpenAI decides on the detail level if auto is specified)
				return image_cfg.low_tokens

		return image_cfg.high_tokens_base + num_tiles * image_cfg.high_tokens_tile

	# Chat completions: Retrieve the resolution of an image URL
	def cc_image_resolution(self, image_url: str) -> Optional[tuple[int, int]]:
		if match := re.fullmatch(r'data:[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126};base64,([A-Za-z0-9+/]+={0,2})', image_url):
			try:
				return PIL.Image.open(io.BytesIO(base64.b64decode(match.group(1), validate=True))).size
			except binascii.Error:
				self.warning("Ignoring input tokens corresponding to image base64 that failed to parse")
			except Exception:  # noqa
				self.warning("Ignoring input tokens corresponding to image base64 that failed to open in PIL")
			return None
		else:
			try:
				response = requests.get(image_url)
				response.raise_for_status()
				return PIL.Image.open(io.BytesIO(response.content)).size
			except requests.RequestException:
				self.warning("Ignoring input tokens corresponding to image URL that could not be retrieved")
			except Exception:  # noqa
				self.warning("Ignoring input tokens corresponding to image URL that failed to open in PIL")
			return None

	# Chat completions: Estimate the number of input tokens required for response format
	def cc_response_format_input_tokens(self, encoding: tiktoken.Encoding, response_format: dict[str, Any], supports_json_schema: bool, model: str, func_cfg: CCFuncTokensConfig) -> int:
		response_format_type = response_format['type']
		if response_format_type == 'text':
			return 0
		elif response_format_type == 'json_object':
			return 0
		elif response_format_type == 'json_schema':
			if not supports_json_schema:
				self.warning(f"Assuming default chat completions function tokens configuration for unrecognized JSON schema response format model: {model}")
			return self.cc_functions_input_tokens(encoding=encoding, functions=(response_format['json_schema'],), function_call=None, func_cfg=func_cfg, key='schema')
		else:
			self.warning(f"Ignoring input tokens corresponding to unrecognized response format type: {response_format_type}")
			return 0

	# Chat completions: Estimate the number of input tokens required for tool use
	def cc_tools_input_tokens(self, encoding: tiktoken.Encoding, tools: Sequence[dict[str, Any]], tool_choice: Union[str, dict[str, Any], None], func_cfg: CCFuncTokensConfig) -> int:

		functions = []
		for tool in tools:
			tool_type = tool['type']
			if tool_type == 'function':
				functions.append(tool['function'])
			else:
				self.warning(f"Ignoring input tokens corresponding to unrecognized tool type: {tool_type}")

		if isinstance(tool_choice, str) or tool_choice is None:
			function_call = tool_choice
		else:
			tool_type = tool_choice['type']
			if tool_type == 'function':
				function_call = tool_choice['function']
			else:
				self.warning(f"Ignoring input tokens corresponding to unrecognized tool choice type: {tool_type}")
				function_call = None

		return self.cc_functions_input_tokens(encoding=encoding, functions=functions, function_call=function_call, func_cfg=func_cfg)

	# Chat completions: Estimate the number of input tokens required for function use
	def cc_functions_input_tokens(self, encoding: tiktoken.Encoding, functions: Sequence[dict[str, Any]], function_call: Union[str, dict[str, Any], None], func_cfg: CCFuncTokensConfig, key: str = 'parameters') -> int:

		if not functions:
			return 0  # Note: It is actually an API error to pass empty tools/functions parameters, so we give it 0 tokens

		total_tokens = 0
		if function_call is None:
			function_call = 'auto'
		if function_call == 'none':
			total_tokens += func_cfg.call_none

		for function in functions:

			total_tokens += func_cfg.func_init + len(encoding.encode(function['name']))

			function_desc: str = function.get('description', '')
			if function_desc:
				total_tokens += func_cfg.func_desc_pre
			function_desc = function_desc.strip()
			if function_desc:
				total_tokens += func_cfg.func_desc_post
			total_tokens += len(encoding.encode(re.sub(rf'\.{{0,{func_cfg.desc_strip_dot}}}$', r'', function_desc)))

			function_params: Optional[dict[str, Any]] = function.get(key, None)
			if function_params:
				function_params_type: Optional[str] = function_params.get('type', None)
				if function_params_type != 'object':
					self.warning(f"Ignoring input tokens corresponding to functions due to unrecognized root function parameters type: {function_params_type}")
				else:
					function_title: Optional[str] = function_params.get('title', None)
					function_defs: Optional[dict[str, Any]] = function_params.get('$defs', None)
					function_props: Optional[dict[str, Any]] = function_params.get('properties', None)
					if function_title:
						total_tokens += func_cfg.prop_title_init + len(encoding.encode(function_title))
					if function_defs or function_props:
						total_tokens += func_cfg.root_props_init
					if function_defs:
						total_tokens += self.cc_object_props_input_tokens(encoding=encoding, props=function_defs, func_cfg=func_cfg)
					if function_props:
						total_tokens += self.cc_object_props_input_tokens(encoding=encoding, props=function_props, func_cfg=func_cfg)

		total_tokens += func_cfg.func_end

		return total_tokens

	# Chat completions: Estimate the number of input tokens required for a particular set of object properties as part of function use
	def cc_object_props_input_tokens(self, encoding: tiktoken.Encoding, props: dict[str, Any], func_cfg: CCFuncTokensConfig) -> int:
		total_tokens = 0
		last_valid = None
		for prop_key, prop_spec in props.items():
			last_valid = None
			prop_type = prop_spec.get('type', None)
			prop_title = prop_spec.get('title', None)
			prop_desc = prop_spec.get('description', None)
			prop_props = prop_spec.get('properties', None)
			prop_items = prop_spec.get('items', None)
			if prop_type is None:
				self.warning("Ignoring input tokens corresponding to type-less property in function schema")
			elif prop_type == 'object' and prop_props is None:
				self.warning("Ignoring input tokens corresponding to properties-less object in function schema")
			elif prop_type == 'array' and prop_items is None:
				self.warning("Ignoring input tokens corresponding to items-less array in function schema")
			elif prop_type == 'array' and prop_items and 'type' not in prop_items:
				self.warning("Ignoring input tokens corresponding to a type-less array")
			elif prop_type == 'array' and prop_items.get('type', None) == 'object' and 'properties' not in prop_items:
				self.warning("Ignoring input tokens corresponding to array of properties-less objects in function schema")
			else:
				total_tokens += func_cfg.prop_key_init + len(encoding.encode(prop_key))
				total_tokens += func_cfg.prop_title_init + len(encoding.encode(prop_title))
				if prop_type == 'object':
					total_tokens += self.cc_description_input_tokens(encoding=encoding, desc=prop_desc, func_cfg=func_cfg, prop_desc=func_cfg.prop_desc_obj)
					if prop_props:
						last_valid = 'object'
						total_tokens += self.cc_object_props_input_tokens(encoding=encoding, props=prop_props, func_cfg=func_cfg)
					else:
						total_tokens += func_cfg.prop_special_obj
				elif prop_type == 'array':
					last_valid = 'array'
					if not prop_items:
						prop_items = {'type': 'object', 'properties': {}}
					array_type = prop_items['type']
					total_tokens += len(encoding.encode(array_type))
					total_tokens += self.cc_description_input_tokens(encoding=encoding, desc=prop_desc, func_cfg=func_cfg, prop_desc=func_cfg.prop_desc_oth)
					prop_items_desc = prop_items.get('description', None)
					if array_type == 'object' and prop_items['properties']:
						total_tokens += self.cc_object_props_input_tokens(encoding=encoding, props=prop_items['properties'], func_cfg=func_cfg)
						total_tokens += self.cc_description_input_tokens(encoding=encoding, desc=prop_items_desc, func_cfg=func_cfg, prop_desc=func_cfg.prop_desc_item_obj)
					else:
						total_tokens += self.cc_description_input_tokens(encoding=encoding, desc=prop_items_desc, func_cfg=func_cfg, prop_desc=func_cfg.prop_desc_item_oth)
				else:
					total_tokens += len(encoding.encode(prop_type))
					prop_enum = prop_spec.get('enum', None)
					if prop_enum:
						last_valid = 'enum'
						total_tokens += (len(prop_enum) - 1) * func_cfg.prop_enum_gap + sum(len(encoding.encode(prop_enum_value)) for prop_enum_value in prop_enum)
						total_tokens += self.cc_description_input_tokens(encoding=encoding, desc=prop_desc, func_cfg=func_cfg, prop_desc=func_cfg.prop_desc_oth)
					else:
						total_tokens += self.cc_description_input_tokens(encoding=encoding, desc=prop_desc, func_cfg=func_cfg, prop_desc=func_cfg.prop_desc_value)
		if last_valid == 'array' or last_valid == 'enum':
			total_tokens += func_cfg.prop_special_last
		elif last_valid == 'object':
			total_tokens += func_cfg.prop_special_obj
		return total_tokens

	# Chat completions: Estimate the number of input tokens required for a description as part of function use
	@classmethod
	def cc_description_input_tokens(cls, encoding: tiktoken.Encoding, desc: Optional[str], func_cfg: CCFuncTokensConfig, prop_desc: int) -> int:
		if desc:
			desc = desc.strip()
			if desc:
				return prop_desc + len(encoding.encode(re.sub(rf'\.{{0,{func_cfg.desc_strip_dot}}}$', r'', desc)))
		return 0

	# Roughly approximate the number of output tokens a request payload will have based on the maximum number of tokens allowed
	def payload_output_tokens(self, payload: dict[str, Any], endpoint: str) -> tuple[int, int]:
		if endpoint == '/v1/chat/completions':  # Chat completions endpoint
			max_output_tokens = max(payload.get('max_completion_tokens', payload.get('max_tokens', 2048)), 0)
			output_tokens = round(max_output_tokens * self.assumed_completion_ratio)
		else:
			raise ValueError(f"Cannot estimate output tokens for unrecognized endpoint: {endpoint}")
		return output_tokens, max_output_tokens

# Token coster class
class TokenCoster:

	def __init__(self, cost_input_direct_mtoken: float, cost_input_cached_mtoken: float, cost_input_batch_mtoken: float, cost_output_direct_mtoken: float, cost_output_batch_mtoken: float):
		if cost_input_direct_mtoken < 0 or cost_input_cached_mtoken < 0 or cost_input_batch_mtoken < 0 or cost_output_direct_mtoken < 0 or cost_output_batch_mtoken < 0:
			raise ValueError(f"Costs cannot be negative: {cost_input_direct_mtoken:.3f}, {cost_input_cached_mtoken:.3f}, {cost_input_batch_mtoken:.3f}, {cost_output_direct_mtoken:.3f}, {cost_output_batch_mtoken:.3f}")
		self.cost_input_direct = cost_input_direct_mtoken / 1e6
		self.cost_input_cached = cost_input_cached_mtoken / 1e6
		self.cost_input_batch = cost_input_batch_mtoken / 1e6
		self.cost_output_direct = cost_output_direct_mtoken / 1e6
		self.cost_output_batch = cost_output_batch_mtoken / 1e6

	def input_cost(self, direct: int = 0, cached: int = 0, batch: int = 0) -> float:
		return direct * self.cost_input_direct + cached * self.cost_input_cached + batch * self.cost_input_batch

	def output_cost(self, direct: int = 0, batch: int = 0) -> float:
		return direct * self.cost_output_direct + batch * self.cost_output_batch

	def cost(self, input_direct: int = 0, input_cached: int = 0, input_batch: int = 0, output_direct: int = 0, output_batch: int = 0) -> float:
		return self.input_cost(direct=input_direct, cached=input_cached, batch=input_batch) + self.output_cost(direct=output_direct, batch=output_batch)
# EOF
