# Demonstrate the task manager module

# Imports
from __future__ import annotations
import os
import sys
import enum
import logging
import argparse
import functools
from typing import Sequence, Optional, Any
import pydantic
from . import gpt_requester, task_manager, utils

# TODO: Add any new demos to gpt_batch_api/commands.txt
# TODO: Demo a basic single opinion/single stage text writing task with simple JSON output
# TODO: Demo a multi-opinion task
# TODO: Demo a multi-stage task (continuing the conversation after first response)
# TODO: One demo should have chunked JSONL output (size- and/or line-limited possible)
# TODO: At least one demo should use structured outputs, and at least one shouldn't
# TODO: Direct is tested by having an auto-direct thing when an odd small number of samples left at the end (make sure the numbers are such that a small thing is left over)

# TODO: Standardise structured output error handling (refusals, stop reason is length (in all cases), etc)
# TODO: Retrying: Can it be lower, inside GPTRequester? => Means you also don't have to rebuild the payload
# TODO: When response comes, can attempt to parse, and return a bool saying RETRY if the exact same payload should be retried => If NOT retry, then it has to end up in failed_samples or succeeded_samples
# TODO: State what the unicode character is, give a one sentence description of what the symbol represents and where it comes from, give a sample sentence including the character at least once (if not included then this is a 'parse error' situation, i.e. soft fail)
# TODO: _task.json file for storing basically information about what you've already processed (must be hashable after lists are converted to tuples and dicts are converted to tuples of items?)
# TODO: _output.json OR _data_001.jsonl file for storing the ACTUAL output/processed response data (NOT needed to determine what samples are needed in future)

#
# Demo: Character codes
#

# Character codes task class
class CharCodesTask(task_manager.TaskManager):

	class UnicodeCharacterType(str, enum.Enum):
		LETTER = "letter"            # Any alphabetic character
		NUMBER = "number"            # Numeric characters
		PUNCTUATION = "punctuation"  # Characters like commas, periods, and such
		SYMBOL = "symbol"            # Mathematical or other symbols
		CURRENCY = "currency"        # Currency symbols
		CONTROL = "control"          # Control characters
		SPACE = "space"              # Whitespace characters
		MARK = "mark"                # Combining marks
		EMOJI = "emoji"              # Emoji characters
		OTHER = "other"              # Any other type not covered by the above categories

	class UnicodeCharacterInfo(pydantic.BaseModel):
		character: str = pydantic.Field(title="Unicode character", description="The unicode character in question (a string containing only the single literal character).")
		type: CharCodesTask.UnicodeCharacterType = pydantic.Field(title="Character type", description="The best-matching type of the unicode character.")
		description: str = pydantic.Field(title="Character description", description="A one-sentence description of what the character symbol represents and where it comes from.")
		sample_sentence: str = pydantic.Field(title="Sample sentence", description="A sample sentence including the character at least twice (as part of some of the words in the sentence).")

	def __init__(self, cfg: utils.Config, task_dir: str, char_ranges: Sequence[tuple[int, int]]):
		gpt_requester_kwargs = gpt_requester.GPTRequester.get_kwargs(cfg)
		gpt_requester_kwargs.update(endpoint=cfg.chat_endpoint)
		super().__init__(
			task_dir=task_dir,
			name_prefix=cfg.task_prefix,
			reinit_meta=cfg.reinit_meta,
			init_meta=dict(  # Note: init_meta specifies parameter values that should always remain fixed throughout a task, even across multiple runs (this behaviour can be manually overridden using reinit_meta)
				model=resolve(cfg.model, default='gpt-4o-mini-2024-07-18'),
				max_completion_tokens=resolve(cfg.max_completion_tokens, default=384),
				temperature=resolve(cfg.temperature, default=0.2),
				top_p=resolve(cfg.top_p, default=0.6),
			),
			**gpt_requester_kwargs,
		)
		self.cfg = cfg  # Note: self.cfg is the source for parameter values that should always be taken from the current run (amongst other parameters)
		self.char_ranges = char_ranges

	def generate_requests(self) -> bool:

		for char_range in self.char_ranges:

			for char_code_point in range(char_range[0], char_range[1] + 1):
				char = chr(char_code_point)
				if char.isprintable():
					sample_key = f'char-{char}'
					if sample_key not in self.T.committed_samples:
						self.GR.add_request(gpt_requester.GPTRequest(
							payload=dict(
								model=self.T.meta['model'],
								max_completion_tokens=self.T.meta['max_completion_tokens'],
								temperature=self.T.meta['temperature'],
								top_p=self.T.meta['top_p'],
								messages=[
									dict(role='system', content="Given a unicode character, provide information about it."),
									dict(role='user', content=f"Character: \"{char}\"" if char.isspace() else f"Character: {char}"),
								],
								response_format=CharCodesTask.UnicodeCharacterInfo,
							),
							meta=dict(
								sample_key=sample_key,
								char=char,
							),
						))

			if self.GR.P and self.commit_requests():
				return False

		return True

	def commit_cached_request(self, cached_req: gpt_requester.CachedGPTRequest):
		self.T.committed_samples[cached_req.item.req.meta['sample_key']] = None

	def cached_request_keys(self, cached_reqs: list[gpt_requester.CachedGPTRequest]) -> Optional[set[str]]:
		sample_keys = {cached_req.item.req.meta['sample_key'] for cached_req in cached_reqs}
		assert len(sample_keys) == len(cached_reqs)
		return sample_keys

	def process_batch_result(self, result: gpt_requester.BatchResult, rstack: utils.RevertStack):
		pass  # TODO: IMPLEMENT

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
# Miscellaneous
#

# Resolve a default non-None value
def resolve(value: Any, default: Any) -> Any:
	return value if value is not None else default

#
# Run
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

# Main function
def main():

	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.set_name('console')
	stream_handler.setFormatter(ColorFormatter(fmt=ColorFormatter.FMT, datefmt=ColorFormatter.DATEFMT))
	logging.basicConfig(level=logging.INFO, format=ColorFormatter.FMT, handlers=[stream_handler])

	parser = argparse.ArgumentParser(description="Demonstrate the TaskManager class with example applications.", add_help=False, formatter_class=functools.partial(argparse.HelpFormatter, max_help_position=36))
	parser.add_argument('--help', '-h', action='help', default=argparse.SUPPRESS, help='Show this help message and exit')
	parser.add_argument('--task', type=str, required=True, help='Which task to run (e.g. char_codes)')
	parser.add_argument('--task_prefix', type=str, metavar='PREFIX', help='Name prefix to use for task-related files (default is same as task)')
	parser.add_argument('--chat_endpoint', type=str, metavar='ENDPOINT', default='/v1/chat/completions', help='Chat completions endpoint to use')

	parser_meta = parser.add_argument_group(title='Task metadata', description='Specifications of the task metadata to be used for new tasks (the default values are defined per-task in the corresponding task implementations).')
	parser_meta.add_argument('--reinit_meta', action='store_true', help="Force reinitialisation of the task metadata for an existing task (normally the task metadata arguments in this group are only used for initialisation of a new task and remain fixed after that across all future runs)")
	parser_meta.add_argument('--model', type=str, help="LLM model to use")
	parser_meta.add_argument('--max_completion_tokens', type=int, metavar='NUM', help="Maximum number of generated output tokens per request (including both reasoning and visible tokens)")
	parser_meta.add_argument('--temperature', type=float, metavar='TEMP', help="What sampling temperature to use")
	parser_meta.add_argument('--top_p', type=float, metavar='MASS', help="Nucleus sampling probability mass")

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
