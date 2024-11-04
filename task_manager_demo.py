# Demonstrate the task manager module

# Imports
import os
import sys
import logging
import argparse
from typing import Sequence
from . import gpt_requester, task_manager, utils

# TODO: Add any new demos to gpt_batch_api/commands.txt
# TODO: Demo a basic single opinion/single stage text writing task with simple JSON output
# TODO: Demo a multi-opinion task
# TODO: Demo a multi-stage task (continuing the conversation after first response)
# TODO: One demo should have chunked JSONL output (size- and/or line-limited possible)
# TODO: At least one demo should use structured outputs, and at least one shouldn't
# TODO: Direct is tested by having an auto-direct thing when an odd small number of samples left at the end (make sure the numbers are such that a small thing is left over)

#
# Demo: Character codes
#

# Character codes task class
class CharCodesTask(task_manager.TaskManager):

	def __init__(self, cfg: utils.Config, task_dir: str, char_ranges: Sequence[tuple[int, int]]):
		super().__init__(
			task_dir=task_dir,
			name_prefix=cfg.task_prefix,
			init_meta=dict(model=cfg.model),
			**gpt_requester.GPTRequester.get_kwargs(cfg),
		)
		self.char_ranges = char_ranges  # TODO: Recall that you still want to ignore non-printable characters: char = chr(code_point); if char.isprintable(): PROCESS sample
		# # TODO: Recall that ultimately you should only add requests for samples (code points) that haven't been committed yet! UNLESS they also completed and FAILED => How to deal with that??




	# # TODO: State what the unicode character is, give a one sentence description of what the symbol represents and where it comes from, give a sample sentence including the character at least once (if not included then this is a 'parse error' situation, i.e. soft fail)
	#
	# # Create a GPT requester to manage getting information about unicode characters
	#
	# # TODO: _task.json file for storing basically information about what you've already processed (must be hashable after lists are converted to tuples and dicts are converted to tuples of items?)
	# # TODO: _output.json OR _data_001.jsonl file for storing the ACTUAL output/processed response data (NOT needed to determine what samples are needed in future)
	#
	# # Enter the GPT requester and thereby allow it to obtain a lock on the state files
	# with gpt_requester:
	# 	# Hard-code the unicode characters we are interested in getting information about (we restrict these ranges further to printable characters)
	#
	# 	# TODO: Sample key should uniquely identify the VALUE of the sample in some explicit/implicit way (so that if I change my set of noun candidates, it just automatically does whichever ones are new and leaves everything else untouched)
	#
	# 	# TODO: DOC, check which actually still need to be done
	# 	code_point_a, code_point_b = unicode_blocks[0]
	# 	for code_point, char in ((cp, chr(cp)) for cp in range(code_point_a, code_point_b + 1) if chr(cp).isprintable()):
	# 		response = gpt_requester.direct_request(TODO)




# Demonstrate the task manager class on the task of generating information about unicode characters
def demo_char_codes(cfg: utils.Config, task_dir: str):
	CharCodesTask(
		cfg=cfg,
		task_dir=task_dir,
		char_ranges=(
			(0x0000, 0x007F),  # Basic Latin
			(0x0080, 0x00FF),  # Latin-1 Supplement
			(0x0100, 0x017F),  # Latin Extended-A
			(0x0180, 0x024F),  # Latin Extended-B
			(0x0250, 0x02AF),  # IPA Extensions
			(0x02B0, 0x02FF),  # Spacing Modifier Letters
			(0x0370, 0x03FF),  # Greek and Coptic
			(0x0400, 0x04FF),  # Cyrillic
			(0x0500, 0x052F),  # Cyrillic Supplement
		),
	).run()

#
# Run
#

# Main function
def main():

	logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(asctime)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

	parser = argparse.ArgumentParser(description="Demonstrate the TaskManager class with example applications.")
	parser.add_argument('--task', type=str, required=True, help='Which task to run (e.g. char_codes)')
	parser.add_argument('--task_prefix', type=str, help='Name prefix to use for task-related files')

	parser_common = parser.add_argument_group('Common')
	parser_common.add_argument('--model', type=str, default='gpt-4o-mini', help="Model to use for new tasks")

	gpt_requester.GPTRequester.configure_argparse(parser=parser)
	args = parser.parse_args()
	if args.task_prefix is None:
		args.task_prefix = args.task

	task_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks')
	if args.task == 'char_codes':
		demo_char_codes(cfg=args, task_dir=task_dir)
	elif args.task is None:
		raise ValueError("Please specify which task to demo using --task")
	else:
		raise ValueError(f"Unrecognised task: {args.task}")

# Run main function
if __name__ == "__main__":
	main()
# EOF
