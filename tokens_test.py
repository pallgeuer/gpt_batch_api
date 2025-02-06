# Test the tokens module

# Imports
import io
import sys
import json
import base64
import logging
from typing import Any
import PIL.Image
import openai
from .logger import log
from . import tokens, utils

#
# Test chat completions
#

# Test input token calculation for chat completions endpoint
def test_chat_completions(client: openai.OpenAI, token_est: tokens.TokenEstimator):

	def test_token_est(task_: dict[str, Any]):
		log.info('\u2500' * 120)
		log.info(f"REQUEST:\n{json.dumps(task_, ensure_ascii=False, indent=2)}")
		expected_tokens = token_est.payload_input_tokens(payload=task_, endpoint='/v1/chat/completions')
		try:
			response = client.chat.completions.create(**task_)
		except openai.OpenAIError as e:
			log.error(f"\033[31mINPUT TOKENS: Expected {expected_tokens} and got {utils.get_class_str(e)}: {e}\033[0m")
		else:
			if expected_tokens.total == response.usage.prompt_tokens:
				log.info(f"\033[32mINPUT TOKENS: Expected {expected_tokens} and got {response.usage.prompt_tokens}\033[0m")
			else:
				log.warning(f"\033[33mINPUT TOKENS: Expected {expected_tokens} and got {response.usage.prompt_tokens}\033[0m")

	for model in ('gpt-3.5-turbo', 'gpt-4', 'o1-preview', 'o1-mini'):

		base_task = dict(
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
			{**base_task, 'messages': [{**base_task['messages'][0], 'content': base_task['messages'][0]['content'][:1]}]},
			{**base_task, 'messages': [{**base_task['messages'][0], 'content': base_task['messages'][0]['content'][:2]}]},
			{**base_task, 'messages': [{**base_task['messages'][0], 'content': base_task['messages'][0]['content'][:3]}]},
			{**base_task, 'messages': base_task['messages'][:2]},
			{**base_task, 'messages': base_task['messages'][:3]},
			{**base_task, 'messages': base_task['messages'][:4]},
		):
			test_token_est(task)

	black_image = PIL.Image.new(mode='RGB', size=(16, 16))
	black_image.save((buffered := io.BytesIO()), format="PNG")
	black_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

	for model in ('gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4o', 'gpt-4o-mini'):

		base_task = dict(
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
			max_completion_tokens=1,
		)

		for task in (
			{**base_task, 'messages': base_task['messages'][:1]},
			{**base_task, 'messages': [base_task['messages'][0], {**base_task['messages'][1], 'content': base_task['messages'][1]['content'][:1]}]},
			{**base_task, 'messages': [base_task['messages'][0], {**base_task['messages'][1], 'content': base_task['messages'][1]['content'][:2]}]},
			{**base_task, 'messages': [base_task['messages'][0], {**base_task['messages'][1], 'content': base_task['messages'][1]['content'][:3]}]},
			{**base_task, 'messages': [base_task['messages'][0], {**base_task['messages'][1], 'content': base_task['messages'][1]['content'][:4]}]},
			{**base_task, 'messages': [base_task['messages'][0], {**base_task['messages'][1], 'content': base_task['messages'][1]['content'][:5]}]},
			{**base_task, 'messages': base_task['messages'][:3]},
			{**base_task, 'messages': base_task['messages'][:4]},
		):
			test_token_est(task)

#
# Run
#

# Main function
def main():

	logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(asctime)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
	client = openai.OpenAI()
	token_est = tokens.TokenEstimator(warn='always', assumed_completion_ratio=0.5)

	test_chat_completions(client=client, token_est=token_est)

# Run main function
if __name__ == "__main__":
	main()
# EOF
