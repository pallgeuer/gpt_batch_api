# Demonstrate the task manager module

# Imports
import os
import sys
import logging
import argparse
from . import task_manager

# TODO: Add any new demos to gpt_batch_api/commands.txt
# TODO: Demo a basic single opinion/single stage text writing task with simple JSON output
# TODO: Demo a multi-opinion task
# TODO: Demo a multi-stage task (continuing the conversation after first response)
# TODO: One demo should have chunked JSONL output

#
# Demo: Character codes
#

# Demonstrate the task manager class on the task of generating information about unicode characters
def demo_char_codes():

	# TODO: State what the unicode character is, give a one sentence description of what the symbol represents and where it comes from, give a sample sentence including the character at least once (if not included then this is a 'parse error' situation, i.e. soft fail)
	# TODO: Only allow printable/new characters

	# Create a GPT requester to manage getting information about unicode characters
	working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks')
	gpt_requester = GPTRequester(working_dir=working_dir, name_prefix='demo_char_codes')

	# TODO: _task.json file for storing basically information about what you've already processed (must be hashable after lists are converted to tuples and dicts are converted to tuples of items?)
	# TODO: _output.json OR _data_001.jsonl file for storing the ACTUAL output/processed response data (NOT needed to determine what samples are needed in future)

	# Enter the GPT requester and thereby allow it to obtain a lock on the state files
	with gpt_requester:
		# Hard-code the unicode characters we are interested in getting information about (we restrict these ranges further to printable characters)
		unicode_blocks = (
			(0x0000, 0x007F),  # Basic Latin
			(0x0080, 0x00FF),  # Latin-1 Supplement
			(0x0100, 0x017F),  # Latin Extended-A
			(0x0180, 0x024F),  # Latin Extended-B
			(0x0250, 0x02AF),  # IPA Extensions
			(0x02B0, 0x02FF),  # Spacing Modifier Letters
			(0x0370, 0x03FF),  # Greek and Coptic
			(0x0400, 0x04FF),  # Cyrillic
			(0x0500, 0x052F),  # Cyrillic Supplement
		)

		# TODO: DOC, check which actually still need to be done
		code_point_a, code_point_b = unicode_blocks[0]
		for code_point, char in ((cp, chr(cp)) for cp in range(code_point_a, code_point_b + 1) if chr(cp).isprintable()):
			response = gpt_requester.direct_request(TODO)

	# TODO: Direct
	# TODO: add_request
	# TODO: add_requests
	# TODO: add+commit simultaneous
	pass  # TODO for code_point in range(0x0300): char = chr(code_point); if char.isprintable(): PROCESS sample

#
# Run
#

# Main function
def main():

	logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(asctime)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

	parser = argparse.ArgumentParser(description="Demonstrate the TaskManager class with example applications")
	parser.add_argument('--task', type=str, help='Demo task to run')
	# TODO: Demonstrate default argparse use to supply configs (also allow custom default values with fallback to default default-values)
	args = parser.parse_args()

	if args.task == 'char_codes':
		demo_char_codes()
	elif args.task is None:
		raise ValueError("Please specify which task to demo using --task")
	else:
		raise ValueError(f"Unrecognised task: {args.task}")

# Run main function
if __name__ == "__main__":
	main()
# EOF
