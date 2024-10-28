# Utilities for counting and managing tokens

# Imports
import io
import re
import sys
import json
import base64
import string
import logging
import binascii
import collections
import dataclasses
from typing import Any, Optional, Union
import PIL.Image
import requests
import tiktoken
import openai
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
	tools: int          # Number of input tokens for specifying available tools
	functions: int      # Number of input tokens for specifying available functions
	extra: int          # Number of hidden extra input tokens (including fixed)

	total: int          # Total number of input tokens

# Token estimator class
class TokenEstimator:

	def __init__(self, warn: str = 'once'):
		self.warn = warn
		if self.warn not in ('never', 'once', 'always'):
			raise ValueError(f"Invalid warn mode: {self.warn}")
		self.seen_warnings = set()

	def reset(self):
		self.seen_warnings.clear()

	def warning(self, msg: str):
		if self.warn == 'always':
			log.warning(msg)
		elif self.warn == 'once' and msg not in self.seen_warnings:
			log.warning(f"{msg} (WARN ONCE)")
			self.seen_warnings.add(msg)

	# Parse from a JSON payload all recognised content that counts towards input tokens and thereby estimate the input token requirements
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
				self.warning(f"Assuming '{encoding.name}' encoding for unrecognised model: {model}")

			if re.match(r'(ft:)?(?:gpt-3.5|gpt-4|gpt-4-turbo|(?:chat)?gpt-4o(?:-mini)?)', model):
				extra_tokens_per_response = 3
				extra_tokens_per_message = 3
				extra_tokens_per_message_gap = 0
				extra_tokens_last_content_unended = 0
				extra_tokens_per_name = 1
			elif re.match(r'(ft:)?o1', model):
				extra_tokens_per_response = 3
				extra_tokens_per_message = 3
				extra_tokens_per_message_gap = 7
				extra_tokens_last_content_unended = 1
				extra_tokens_per_name = 4
			else:
				self.warning(f"Assuming default extra token counts for unrecognised model: {model}")
				extra_tokens_per_response = 3
				extra_tokens_per_message = 3
				extra_tokens_per_message_gap = 0
				extra_tokens_last_content_unended = 0
				extra_tokens_per_name = 1
			special_end_tokens = (encoding.encode(' ')[0], encoding.encode('.')[0])

			supports_image = True
			if re.match(r'(ft:)?gpt-4o-mini', model):
				image_low_tokens = 2833
				image_high_tokens_base = 2833
				image_high_tokens_tile = 5667
				image_high_res = 768
				image_high_tile = 512
			elif re.match(r'(ft:)?(?:gpt-4-turbo|(?:chat)?gpt-4o)', model):
				image_low_tokens = 85
				image_high_tokens_base = 85
				image_high_tokens_tile = 170
				image_high_res = 768
				image_high_tile = 512
			else:
				image_low_tokens = 85
				image_high_tokens_base = 85
				image_high_tokens_tile = 170
				image_high_res = 768
				image_high_tile = 512
				supports_image = False

			fixed = extra_tokens_per_response
			extra += extra_tokens_per_response
			for m, message in enumerate(payload['messages'], 1):
				role = message['role']
				message_tokens = extra_tokens_per_message
				if m > 1:
					message_tokens += extra_tokens_per_message_gap
				msg_tokens[role] += message_tokens
				extra += message_tokens
				for key, value in message.items():
					if key == 'content':
						if isinstance(value, str):
							value_tokens = len(encoding.encode(value))
							if value and value[-1] not in string.whitespace and value[-1] not in string.punctuation:
								value_tokens += extra_tokens_last_content_unended
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
										self.warning(f"Assuming default image token counts for unrecognised image model: {model}")
									image_spec = content_item['image_url']
									image_tokens = self.image_input_tokens(
										url=image_spec['url'],
										detail=image_spec.get('detail', 'auto'),
										low_tokens=image_low_tokens,
										high_res=image_high_res,
										high_tile=image_high_tile,
										high_tokens_base=image_high_tokens_base,
										high_tokens_tile=image_high_tokens_tile,
									)
									msg_tokens[role] += image_tokens
									type_tokens['image'] += image_tokens
								else:
									self.warning(f"Ignoring input tokens corresponding to unrecognised content type: {content_type}")
								last_content_type = content_type
							if content_type == 'text' and content_text and content_text[-1] not in string.whitespace and content_text[-1] not in string.punctuation:
								msg_tokens[role] += extra_tokens_last_content_unended
								type_tokens['text'] += extra_tokens_last_content_unended
					elif isinstance(value, str):
						if key == 'name':
							msg_tokens[role] += extra_tokens_per_name
							extra += extra_tokens_per_name
						value_tokens = len(encoding.encode(value))
						msg_tokens[role] += value_tokens
						meta += value_tokens

		else:
			raise ValueError(f"Cannot estimate input tokens for unrecognised endpoint: {endpoint}")

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
		tools = type_tokens['tools']
		functions = type_tokens['functions']

		total = fixed + msg_total + tools + functions
		assert meta + text + image + tools + functions + extra == total

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
			tools=tools,
			functions=functions,
			extra=extra,
			total=total,
		)

	# Estimate the number of input tokens required for an image
	# Helpful source (accessed 28/10/2024): https://platform.openai.com/docs/guides/vision/calculating-costs
	def image_input_tokens(self, url: Union[str, tuple[int, int], None], detail: str, low_tokens: int, high_res: int, high_tile: int, high_tokens_base: int, high_tokens_tile: int) -> int:

		if detail == 'low':
			return low_tokens

		image_res = self.image_resolution(url) if isinstance(url, str) else url
		if image_res is None:
			return 0  # Ignore image if failed to identify its resolution
		width, height = image_res
		if width <= 0 or height <= 0:
			self.warning("Ignoring input tokens corresponding to image with non-positive resolution")
			return 0  # Ignore image if identified resolution has non-positive entries

		if min(width, height) > high_res:
			scale = high_res / min(width, height)
			width = round(width * scale)
			height = round(height * scale)

		num_tiles = ((width - 1) // high_tile + 1) * ((height - 1) // high_tile + 1)
		if detail == 'auto':
			self.warning("Cannot accurately determine number of input image tokens if detail level is 'auto' => Explicitly specify high/low if possible")
			if num_tiles <= 1:  # Assume low detail if resolution fits in a single tile (it is unclear how OpenAI decides on the detail level if auto is specified)
				return low_tokens

		return high_tokens_base + num_tiles * high_tokens_tile

	# Retrieve the resolution of an image URL
	def image_resolution(self, image_url: str) -> Optional[tuple[int, int]]:
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

# TODO: Count tools tokens properly (no crash on extra fields)
# TODO: Count functions tokens properly (no crash on extra fields)
# TODO: Support /v1/embeddings
# TODO: In GPT requester: WARN if input token estimations are far off individually, or definitely too low on average (+2%)

#
# Test
#

# Test input token calculation for chat completions endpoint
def test_chat_completions(client: openai.OpenAI, token_est: TokenEstimator):

	black_image = PIL.Image.new(mode='RGB', size=(16, 16))
	black_image.save((buffered := io.BytesIO()), format="PNG")
	black_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

	def test_token_est(task_: dict[str, Any]):
		log.info('-' * 120)
		log.info(f"REQUEST:\n{json.dumps(task_, indent=2)}")
		expected_tokens = token_est.payload_input_tokens(payload=task_, endpoint='/v1/chat/completions')
		response = client.chat.completions.create(**task_)
		if expected_tokens.total == response.usage.prompt_tokens:
			log.info(f"INPUT TOKENS: Expected {expected_tokens} and got {response.usage.prompt_tokens}")
		else:
			log.warning(f"\033[31mINPUT TOKENS: Expected {expected_tokens} and got {response.usage.prompt_tokens}\033[0m")

	for model in ('gpt-3.5-turbo', 'gpt-4', 'o1-preview', 'o1-mini'):

		base_task_A = dict(
			model=model,
			messages=[
				{
					'role': 'user',
					'content': [
						{'type': 'text', 'text': "I found this image on the internet."},
						{'type': 'text', 'text': "But I can't actually show it to you because you don't understand images"},
						{'type': 'text', 'text': "Right?"},
					],
					'name': 'in-parts',
				},
				{
					'role': 'assistant',
					'content': 'Yes, that is right',
					'name': 'non-image-assistant',
				},
				{
					'role': 'user',
					'content': 'Thought so...',
					'name': 'my-user',
				},
				{
					'role': 'user',
					'content': 'I really thought so?',
				},
			],
			max_completion_tokens=10,
		)

		for task in (
			{**base_task_A, 'messages': [{**base_task_A['messages'][0], 'content': base_task_A['messages'][0]['content'][:1]}]},
			{**base_task_A, 'messages': [{**base_task_A['messages'][0], 'content': base_task_A['messages'][0]['content'][:2]}]},
			{**base_task_A, 'messages': [{**base_task_A['messages'][0], 'content': base_task_A['messages'][0]['content'][:3]}]},
			{**base_task_A, 'messages': base_task_A['messages'][:2]},
			{**base_task_A, 'messages': base_task_A['messages'][:3]},
			{**base_task_A, 'messages': base_task_A['messages'][:4]},
		):
			test_token_est(task)

	for model in ('chatgpt-4o-latest', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'):

		base_task_A = dict(
			model=model,
			messages=[
				{
					'role': 'system',
					'content': 'You are a helpful assistant that can analyze images and provide detailed descriptions.',
				},
				{
					'role': 'user',
					'content': [
						{'type': 'text', 'text': "I found this image on the internet."},
						{'type': 'text', 'text': "Ignore the following image of a black square:"},
						{'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{black_image_base64}", 'detail': 'high'}},
						{'type': 'text', 'text': "What's in this image?"},
						{'type': 'image_url', 'image_url': {'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg', 'detail': 'high'}},
					],
					'name': 'some-user-person',
				},
				{
					'role': 'assistant',
					'content': 'The image shows a scenic boardwalk surrounded by lush greenery and a calm body of water, likely in a nature reserve or park.',
					'name': 'my-helpful-assistant',
				},
				{
					'role': 'user',
					'content': 'Can you provide more details about this place and its significance?',
				},
			],
			max_tokens=1,
		)

		for task in (
			{**base_task_A, 'messages': base_task_A['messages'][:1]},
			{**base_task_A, 'messages': [base_task_A['messages'][0], {**base_task_A['messages'][1], 'content': base_task_A['messages'][1]['content'][:1]}]},
			{**base_task_A, 'messages': [base_task_A['messages'][0], {**base_task_A['messages'][1], 'content': base_task_A['messages'][1]['content'][:2]}]},
			{**base_task_A, 'messages': [base_task_A['messages'][0], {**base_task_A['messages'][1], 'content': base_task_A['messages'][1]['content'][:3]}]},
			{**base_task_A, 'messages': [base_task_A['messages'][0], {**base_task_A['messages'][1], 'content': base_task_A['messages'][1]['content'][:4]}]},
			{**base_task_A, 'messages': [base_task_A['messages'][0], {**base_task_A['messages'][1], 'content': base_task_A['messages'][1]['content'][:5]}]},
			{**base_task_A, 'messages': base_task_A['messages'][:3]},
			{**base_task_A, 'messages': base_task_A['messages'][:4]},
		):
			test_token_est(task)

#
# Run
#

# Main function
def main():

	logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(asctime)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

	client = openai.OpenAI()
	token_est = TokenEstimator(warn='always')

	test_chat_completions(client=client, token_est=token_est)

# Run main function
if __name__ == "__main__":
	main()
# EOF
