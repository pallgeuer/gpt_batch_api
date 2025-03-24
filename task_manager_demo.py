# Demonstrate the task manager module

# Imports
from __future__ import annotations
import os
import enum
import argparse
import functools
import dataclasses
from typing import Sequence, Optional, Any
import pydantic
import openai.types.chat as openai_chat
from . import gpt_requester, task_manager, utils

#
# Demo: Character codes
#

# Unicode character type enumeration (part of the structured output schema used by the GPT)
class UnicodeCharacterType(str, enum.Enum):
	LETTER = 'letter'            # Any alphabetic character
	NUMBER = 'number'            # Numeric characters
	PUNCTUATION = 'punctuation'  # Characters like commas, periods, and such
	SYMBOL = 'symbol'            # Mathematical or other symbols
	CURRENCY = 'currency'        # Currency symbols
	CONTROL = 'control'          # Control characters
	SPACE = 'space'              # Whitespace characters
	MARK = 'mark'                # Combining marks
	EMOJI = 'emoji'              # Emoji characters
	OTHER = 'other'              # Any other type not covered by the above categories

# Unicode character information class (structured output schema used by the GPT)
class UnicodeCharacterInfo(pydantic.BaseModel):
	model_config = pydantic.ConfigDict(strict=True)
	character: str = pydantic.Field(title='Unicode character', description="The unicode character in question (a string containing only the single literal character).")
	type: UnicodeCharacterType = pydantic.Field(title='Character type', description="The best-matching type of the unicode character.", strict=False)  # Note: Not strict due to str vs enumeration type considerations
	description: str = pydantic.Field(title='Character description', description="A one-sentence description of what the character symbol represents and where it comes from.")
	sample_sentence: str = pydantic.Field(title='Sample sentence', description="A sample sentence including the EXACT case-sensitive unicode character codepoint at least TWICE.")

# Character codes data class (data class used for the output file)
@dataclasses.dataclass
class CharCodesData:
	chars: dict[str, UnicodeCharacterInfo] = dataclasses.field(default_factory=dict)  # Map of all characters to their produced output character information

# Character codes file class (output file class that can also be reused in downstream code to read and parse the generated data)
class CharCodesFile(task_manager.DataclassOutputFile[CharCodesData]):

	Dataclass = CharCodesData

	def pre_save(self, rstack: utils.RevertStack):
		rstack.callback(setattr, self.data, 'chars', self.data.chars)
		self.data.chars = dict(sorted(self.data.chars.items()))

	def log_summary(self, *args: dict[str, Any], rstack: utils.RevertStack, flush: bool = False, **kwargs: Any):
		super().log_summary({
			'Output/num_chars': len(self.data.chars),
		}, *args, rstack=rstack, flush=flush, **kwargs)

	def status_str(self) -> str:
		return f"{len(self.data.chars)} chars"

# Character codes task class (runs the task)
class CharCodesTask(task_manager.TaskManager):

	# Sample key:
	#  - Each sample must correspond to a unique (ideally short) string sample key
	#  - 'char-CHAR' where CHAR is the character as a string of length 1 (see generate_requests() method)
	#
	# Task state:           TaskState       => Stores only the information required to know for which samples requests still need to be generated and committed, as well as which samples have been completed so far, and whatever information is needed for compiling the output entry if this is not based on just a single response at a time
	#  - meta:              dict[str, Any]  => Dictionary of task-level metadata like the model and temperature to use for request generation (see init_meta in __init__() method)
	#  - committed_samples: dict[str, None] => Maps a sample key to None, where a key simply existing signifies that a request has been committed for the corresponding character code (see commit_cached_request() method)
	#  - failed_samples:    dict[str, None] => Maps a sample key to None, where a key simply existing signifies that the committed request for the corresponding character code failed (see process_batch_result() method)
	#  - succeeded_samples: dict[str, None] => Maps a sample key to None, where a key simply existing signifies that the committed request for the corresponding character code succeeded (see process_batch_result() method)
	#
	# Task output:
	#  - Single JSON file with the JSON-serialized contents of an instance of the CharCodesData dataclass
	#  - Contains 'chars', a sorted-on-save dict that maps each character code (string containing a single literal character) to the corresponding UnicodeCharacterInfo (contains information like character type, description and sample sentence)

	output: CharCodesFile

	def __init__(self, cfg: utils.Config, char_ranges: Sequence[tuple[int, int]]):

		super().__init__(
			task_dir=cfg.task_dir,
			name_prefix=cfg.task_prefix,
			output_factory=CharCodesFile.output_factory(),
			init_meta=dict(  # Note: init_meta specifies parameter values that should always remain fixed throughout a task, even across multiple runs (this behavior can be manually overridden using reinit_meta)
				model=utils.resolve(cfg.model, default='gpt-4o-mini-2024-07-18'),
				max_completion_tokens=utils.resolve(cfg.max_completion_tokens, default=200),
				completion_ratio=utils.resolve(cfg.completion_ratio, default=0.35),
				temperature=utils.resolve(cfg.temperature, default=0.2),
				top_p=utils.resolve(cfg.top_p, default=0.6),
			),
			**utils.get_init_kwargs(cls=task_manager.TaskManager, cfg=cfg),
			**utils.get_init_kwargs(cls=gpt_requester.GPTRequester, cfg=cfg, endpoint=cfg.chat_endpoint, assumed_completion_ratio=None)
		)

		self.cfg = cfg  # Note: self.cfg is the source for parameter values that should always be taken from the current run (amongst other parameters)
		self.char_ranges = char_ranges

	def on_task_enter(self):
		self.GR.set_assumed_completion_ratio(self.T.meta['completion_ratio'])

	def wipe_unfinished(self, wipe_failed: bool, rstack: utils.RevertStack) -> bool:
		if wipe_failed:
			sample_keys_keep = self.T.succeeded_samples.keys()
			self.T.failed_samples.clear()
		else:
			sample_keys_keep = self.T.succeeded_samples.keys() | self.T.failed_samples.keys()
		for sample_key in tuple(self.T.committed_samples):
			if sample_key not in sample_keys_keep:
				del self.T.committed_samples[sample_key]
		return False

	def validate_state(self, *, clean: bool):
		super().validate_state(clean=clean)
		if clean:
			if unclean_sample_keys := {sample_key for sample_key in self.T.committed_samples.keys() if (sample_key in self.T.succeeded_samples) == (sample_key in self.T.failed_samples)}:
				raise ValueError(f"Unexpected unclean sample keys: {sorted(unclean_sample_keys)}")

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
								response_format=UnicodeCharacterInfo,
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

	def process_batch_result(self, result: gpt_requester.BatchResult, rstack: utils.RevertStack) -> bool:

		CHOICE = 0  # Note: Coded in this parametric way to show how multi-choice situations could be handled, even though here we always only have a single choice and choose the first one
		sample_keys_succeeded = set()
		sample_keys_failed = set()
		sample_chars = set()

		@rstack.callback
		def revert_sample_state():
			for skey in sample_keys_succeeded:
				self.T.succeeded_samples.pop(skey, None)
			for skey in sample_keys_failed:
				self.T.failed_samples.pop(skey, None)
			for schar in sample_chars:
				self.D.chars.pop(schar, None)

		err_char_mismatch, err_char_count, err_misc_failed = [utils.LogSummarizer(log_fn=log.error, show_msgs=self.GR.show_errors) for _ in range(3)]
		for info in result.info.values():

			sample_key = info.req_info.meta['sample_key']
			if sample_key in self.T.succeeded_samples:
				raise ValueError(f"Sample key '{sample_key}' unexpectedly already exists in succeeded samples task state")
			elif sample_key in self.T.failed_samples:
				raise ValueError(f"Sample key '{sample_key}' unexpectedly already exists in failed samples task state")

			sample_char = info.req_info.meta['char']
			if sample_char in self.D.chars:
				raise ValueError(f"Sample character '{sample_char}' unexpectedly already exists in task output")

			char_info: Optional[UnicodeCharacterInfo] = None
			if info.resp_info is not None:
				resp_payload = info.resp_info.payload
				if isinstance(resp_payload, openai_chat.ParsedChatCompletion):
					choices = resp_payload.choices
					if len(choices) > CHOICE >= 0:
						parsed = choices[CHOICE].message.parsed
						if isinstance(parsed, UnicodeCharacterInfo):
							char_info = parsed

			succeeded = False
			warn_infos_choice = [warn_info for warn_info in info.warn_infos if warn_info.type != 'MessageChoice' or warn_info.data == CHOICE]
			if char_info is not None and info.err_info is None and not warn_infos_choice:
				assert not info.retry and info.retry_counts
				if char_info.character != sample_char and char_info.character.strip() != sample_char:
					info.err_info = gpt_requester.ErrorInfo(fatal=False, type='TaskSpecific', subtype='CharMismatch', data=char_info.character, msg=f"Got character '{char_info.character}' instead of '{sample_char}'")
					err_char_mismatch.log(f"Batch {result.batch.id} request ID {info.req_id} retry {info.req_info.retry_num} had a character mismatch: {info.err_info.msg}")
				elif char_info.sample_sentence.count(sample_char) < 2:
					info.err_info = gpt_requester.ErrorInfo(fatal=False, type='TaskSpecific', subtype='TooFewChar', msg=f"Sample sentence has less than 2 occurrences of the sample char '{sample_char}' => \"{char_info.sample_sentence}\"")
					err_char_count.log(f"Batch {result.batch.id} request ID {info.req_id} retry {info.req_info.retry_num}: {info.err_info.msg}")
				else:
					sample_keys_succeeded.add(sample_key)
					self.T.succeeded_samples[sample_key] = None
					sample_chars.add(sample_char)
					self.D.chars[sample_char] = char_info
					succeeded = True
				if not succeeded:
					self.GR.update_result_retry(info=info)

			if not succeeded and not info.retry:
				if info.err_info is None and not self.GR.dryrun:
					err_misc_failed.log(f"Batch {result.batch.id} request ID {info.req_id} retry {info.req_info.retry_num} got no error yet FAILED")
				sample_keys_failed.add(sample_key)
				self.T.failed_samples[sample_key] = None

		err_char_mismatch.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further character mismatches (total {num_total} occurrences)")
		err_char_count.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further sample sentences with less than 2 occurrences of the sample char (total {num_total} samples)")
		err_misc_failed.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further requests that got no error yet FAILED (total {num_total} occurrences)")

		return bool(sample_keys_succeeded) or bool(sample_keys_failed)

# Demonstrate the task manager class on the task of generating information about unicode characters
def demo_char_codes(cfg: utils.Config):
	# cfg = Configuration parameters
	CharCodesTask(
		cfg=cfg,
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
# Demo: Emotion recognition
#

# Utterance emotion enumeration (part of the structured output schema used by the GPT)
class UtteranceEmotion(str, enum.Enum):
	ANGER = 'anger'
	DISGUST = 'disgust'
	SADNESS = 'sadness'
	JOY = 'joy'
	NEUTRAL = 'neutral'
	SURPRISE = 'surprise'
	FEAR = 'fear'

# Utterance information class (structured output schema used by the GPT)
class UtteranceInfo(pydantic.BaseModel):
	model_config = pydantic.ConfigDict(strict=True)
	emotion: UtteranceEmotion = pydantic.Field(title='Utterance emotion', description="The best-matching emotion classification of the utterance.", strict=False)  # Note: Not strict due to str vs enumeration type considerations
	is_clear: bool = pydantic.Field(title='Is clear emotion', description="Whether the emotion classification of the utterance is very clear, and not in any possible way ambiguous.")

# Utterance data class (data class used for the output file)
@dataclasses.dataclass
class UtteranceData:
	sample_key: str                          # Sample key corresponding to utterance
	utterance: str                           # The utterance in question
	is_clear: bool                           # Whether the emotion classification opinions reached a clear majority conclusion
	emotion: Optional[UtteranceEmotion]      # Utterance emotion (None = No single opinion was successful)
	emotions: dict[UtteranceEmotion, float]  # All utterance emotion classifications with their corresponding confidence score (in descending confidence order)

# Utterances file class (output file class that can also be reused in downstream code to read and parse the generated data)
class UtterancesFile(task_manager.DataclassListOutputFile[UtteranceData]):
	Dataclass = UtteranceData

# Utterance emotion task class (runs the task)
class UtteranceEmotionTask(task_manager.TaskManager):

	# Sample key:
	#  - Each sample must correspond to a unique (ideally short) string sample key
	#  - 'I:UTTERANCE_START' where I is the 1-based index of the utterance and UTTERANCE_START is up to 30 leading characters of the utterance (see generate_requests() method)
	#
	# Task state:           TaskState      => Stores only the information required to know for which samples requests still need to be generated and committed, as well as which samples have been completed so far, and whatever information is needed for compiling the output entry if this is not based on just a single response at a time
	#  - meta:              dict[str, Any] => Dictionary of task-level metadata like the model and temperature to use for request generation (see init_meta in __init__() method)
	#  - committed_samples: dict[str, int] => Maps a sample key to the number of times a request has been committed for it so far (see commit_cached_request() method)
	#  - failed_samples:    dict[str, int] => Maps a sample key to the number of committed requests that have failed for that sample key (see process_batch_result() method)
	#  - succeeded_samples: dict[str, list[dict[str, Any]]] ~ dict[str, list[UtteranceInfo]] => Maps a sample key to a list of JSON-serialized UtteranceInfo opinions, one for each committed request that has succeeded for that sample key (see process_batch_result() method)
	#
	# Task output:
	#  - Single JSONL file where each line is the JSON-serialized contents of an instance of the UtteranceData dataclass
	#  - Each line contains a sample key, corresponding utterance, and information about the emotion classifications

	output: UtterancesFile

	def __init__(self, cfg: utils.Config, utterances: Sequence[str]):

		super().__init__(
			task_dir=cfg.task_dir,
			name_prefix=cfg.task_prefix,
			output_factory=UtterancesFile.output_factory(),
			init_meta=dict(  # Note: init_meta specifies parameter values that should always remain fixed throughout a task, even across multiple runs (this behavior can be manually overridden using reinit_meta)
				model=utils.resolve(cfg.model, default='gpt-4o-mini-2024-07-18'),
				max_completion_tokens=utils.resolve(cfg.max_completion_tokens, default=32),
				completion_ratio=utils.resolve(cfg.completion_ratio, default=0.35),
				temperature=utils.resolve(cfg.temperature, default=0.2),
				top_p=utils.resolve(cfg.top_p, default=0.6),
				opinions_min=utils.resolve(cfg.opinions_min, default=3),
				opinions_max=utils.resolve(cfg.opinions_max, default=5),
				confidence=utils.resolve(cfg.confidence, default=0.78),
			),
			**utils.get_init_kwargs(cls=task_manager.TaskManager, cfg=cfg),
			**utils.get_init_kwargs(cls=gpt_requester.GPTRequester, cfg=cfg, endpoint=cfg.chat_endpoint, assumed_completion_ratio=None)
		)

		self.cfg = cfg  # Note: self.cfg is the source for parameter values that should always be taken from the current run (amongst other parameters)
		self.utterances = utterances
		self.opinion_adviser: Optional[task_manager.OpinionAdviser] = None

	def on_task_enter(self):
		self.GR.set_assumed_completion_ratio(self.T.meta['completion_ratio'])
		self.opinion_adviser = task_manager.OpinionAdviser(opinions_min=self.T.meta['opinions_min'], opinions_max=self.T.meta['opinions_max'], confidence=self.T.meta['confidence'])  # Note: TaskManager does not edit self.T.meta for as long as the task manager is entered, so if this class does not ever modify self.T.meta (which it does not) then it is safe to store the meta values inside self.opinion_adviser without risking becoming out-of-date

	def on_task_exit(self):
		self.opinion_adviser = None

	def wipe_unfinished(self, wipe_failed: bool, rstack: utils.RevertStack) -> bool:
		self.T.committed_samples.clear()
		for sample_key, opinions in self.T.succeeded_samples.items():
			self.T.committed_samples[sample_key] = len(opinions)
		if wipe_failed:
			self.T.failed_samples.clear()
			for entry in self.output.rewrite(rstack=rstack):
				opinions: list[dict[str, Any]] = self.T.succeeded_samples.get(entry.sample_key, [])
				if self.opinion_adviser.get_conclusion(num_failed=0, opinions=(opinion['emotion'] for opinion in opinions))[0]:
					self.output.data.entries.append(entry)
			self.D = self.output.data
		else:
			for sample_key, num_failed in self.T.failed_samples.items():
				self.T.committed_samples[sample_key] = self.T.committed_samples.get(sample_key, 0) + num_failed
		return False

	def validate_state(self, *, clean: bool):
		super().validate_state(clean=clean)
		if clean:
			if unclean_sample_keys := {sample_key for sample_key, num_committed in self.T.committed_samples.items() if (len(self.T.succeeded_samples[sample_key]) if sample_key in self.T.succeeded_samples else 0) + self.T.failed_samples.get(sample_key, 0) != num_committed}:
				raise ValueError(f"Unexpected unclean sample keys: {sorted(unclean_sample_keys)}")
		self.opinion_adviser.validate()

	def generate_requests(self) -> bool:

		for i, utterance in enumerate(self.utterances, 1):

			sample_key = f'{i}:{utterance[:30]}'  # Note: If it is possible that the source utterance list changes (add/remove entries) between runs of the same instance of a task, then {i} is not necessarily consistent for a sample, and should be replaced with a checksum or such instead (or the whole utterance could be the sample key if they are known to never be too long)

			num_committed = self.T.committed_samples.get(sample_key, 0)
			num_failed = self.T.failed_samples.get(sample_key, 0)
			opinions: list[dict[str, Any]] = self.T.succeeded_samples.get(sample_key, [])
			assert num_committed >= num_failed + len(opinions) >= 0 and num_failed >= 0

			num_required = max(0, self.opinion_adviser.min_responses_reqd(num_failed=num_failed, opinions=(opinion['emotion'] for opinion in opinions)) - num_committed)
			if num_required > 0:
				request = gpt_requester.GPTRequest(
					payload=dict(
						model=self.T.meta['model'],
						max_completion_tokens=self.T.meta['max_completion_tokens'],
						temperature=self.T.meta['temperature'],
						top_p=self.T.meta['top_p'],
						messages=[
							dict(role='system', content="Given an utterance, perform emotion recognition by classifying the utterance as one of the MELD dataset emotion categories."),
							dict(role='user', content=f"UTTERANCE:\n{utterance}"),
						],
						response_format=UtteranceInfo,
					),
					meta=dict(
						sample_key=sample_key,
						utterance=utterance,
					),
				)
				self.GR.add_requests(request for _ in range(num_required))

		return True

	def commit_cached_request(self, cached_req: gpt_requester.CachedGPTRequest):
		sample_key = cached_req.item.req.meta['sample_key']
		self.T.committed_samples[sample_key] = self.T.committed_samples.get(sample_key, 0) + 1

	def cached_request_keys(self, cached_reqs: list[gpt_requester.CachedGPTRequest]) -> Optional[set[str]]:
		return {cached_req.item.req.meta['sample_key'] for cached_req in cached_reqs}

	def process_batch_result(self, result: gpt_requester.BatchResult, rstack: utils.RevertStack) -> bool:

		succeeded_samples = {}
		failed_samples = {}
		num_entries = len(self.D.entries)

		@rstack.callback
		def revert_sample_state():
			for skey, value in succeeded_samples.items():
				if value is None:
					del self.T.succeeded_samples[skey]
				else:
					del self.T.succeeded_samples[skey][value:]
			for skey, value in failed_samples.items():
				if value is None:
					del self.T.failed_samples[skey]
				else:
					self.T.failed_samples[skey] = value
			del self.D.entries[num_entries:]

		err_misc_failed = utils.LogSummarizer(log_fn=log.error, show_msgs=self.GR.show_errors)
		for info in result.info.values():

			sample_key = info.req_info.meta['sample_key']
			utterance = info.req_info.meta['utterance']

			num_committed = self.T.committed_samples.get(sample_key, 0)
			num_failed = self.T.failed_samples.get(sample_key, 0)
			opinions: list[dict[str, Any]] = self.T.succeeded_samples.get(sample_key, [])
			assert num_committed > num_failed + len(opinions) >= 0 and num_failed >= 0  # Whether the current result is a success, failure or retry, it will always (eventually) count +1 towards the number of responses, hence why num_committed needs to be STRICTLY greater than the current number of responses (prior to the current result being added)

			if info.err_info is None and info.resp_info is not None and isinstance(info.resp_info.payload, openai_chat.ParsedChatCompletion) and info.resp_info.payload.choices and isinstance((utterance_info := info.resp_info.payload.choices[0].message.parsed), UtteranceInfo):
				assert not info.retry and info.retry_counts
				if sample_key in self.T.succeeded_samples:
					if sample_key not in succeeded_samples:
						succeeded_samples[sample_key] = len(opinions)  # For revert_sample_state(): Make a note to truncate self.T.succeeded_samples[sample_key] back down to the original number of opinions if reverting state updates in future
				else:
					if sample_key not in succeeded_samples:
						succeeded_samples[sample_key] = None  # For revert_sample_state(): Make a note to delete sample_key from self.T.succeeded_samples if reverting state updates in future
					self.T.succeeded_samples[sample_key] = opinions
				opinions.append(utterance_info.model_dump(mode='json'))
			elif not info.retry:
				if info.err_info is None and not self.GR.dryrun:
					err_misc_failed.log(f"Batch {result.batch.id} request ID {info.req_id} retry {info.req_info.retry_num} got no error yet FAILED")
				if sample_key not in failed_samples:
					failed_samples[sample_key] = self.T.failed_samples.get(sample_key, None)  # For revert_sample_state(): Store the original number of failed requests, or None if sample_key originally did not exist in self.T.failed_samples
				num_failed += 1
				self.T.failed_samples[sample_key] = num_failed

			if num_committed == num_failed + len(opinions):
				have_conclusion, is_confident, opinions_counter, most_common_opinion, _ = self.opinion_adviser.get_conclusion(num_failed=num_failed, opinions=(opinion['emotion'] for opinion in opinions))
				if have_conclusion:
					total_clarity = sum(1.0 if opinion['is_clear'] else 0.5 for opinion in opinions)
					emotions = {UtteranceEmotion(emotion): sum(1.0 if opinion['is_clear'] else 0.5 for opinion in opinions if opinion['emotion'] == emotion) / total_clarity for emotion in opinions_counter.keys()}
					self.D.entries.append(UtteranceData(
						sample_key=sample_key,
						utterance=utterance,
						is_clear=is_confident,
						emotion=UtteranceEmotion(most_common_opinion) if most_common_opinion is not None else None,
						emotions=dict(sorted(emotions.items(), key=lambda item: (-item[1], item[0]))),
					))

		err_misc_failed.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further requests that got no error yet FAILED (total {num_total} occurrences)")

		return bool(succeeded_samples) or bool(failed_samples)

# Demonstrate the task manager class on the task of classifying the emotion of utterances
def demo_utterance_emotion(cfg: utils.Config):
	# cfg = Configuration parameters
	UtteranceEmotionTask(
		cfg=cfg,
		utterances=(
			"I can’t believe you went behind my back after everything I’ve done for you!",
			"Why do you always have to ruin every single thing I plan?",
			"Stop interrupting me when I’m trying to explain what happened!",
			"I’ve had enough of your excuses; just admit you messed up!",
			"This is the last time I let you talk to me like that!",
			"I can’t stand the way he chews with his mouth open; it’s revolting.",
			"That’s absolutely disgusting—I can’t believe you would even suggest it.",
			"The way they treated her was so awful, it made me sick to my stomach.",
			"I can’t believe you actually enjoy eating something that smells like this.",
			"That behavior is so gross; I don’t even want to be associated with it.",
			"I don’t think I’ll ever get over how much this hurts.",
			"It feels like no matter what I do, I’m always letting everyone down.",
			"I just really miss the way things used to be when everything felt normal.",
			"I wish I could change the past, but I know that’s not possible.",
			"It’s hard to stay positive when everything keeps falling apart.",
			"I can’t believe it—we actually won the competition!",
			"This is the happiest I’ve felt in such a long time; thank you for making it possible.",
			"The surprise party was amazing; I felt so loved and appreciated.",
			"I’ve been laughing non-stop all day; everything just feels perfect!",
			"I’m so proud of what we’ve accomplished together—it’s incredible.",
			"Let me know if you’re free tomorrow so we can finalize the plans.",
			"I think the meeting starts at 3 p.m., but I’ll double-check the schedule.",
			"Can you send me the details later? I need to make sure I’ve got everything.",
			"It seems like a decent option, but I’d need more information before deciding.",
			"I saw your message and will get back to you once I’ve had a chance to think about it.",
			"Wait, you’re telling me that they finished the project a week early?",
			"I wasn’t expecting to see you here—it’s such a pleasant surprise!",
			"How did you manage to pull this off without me finding out?",
			"I can’t believe it actually worked; I was so sure it wouldn’t!",
			"You’re kidding, right? That’s the last thing I ever thought would happen!",
			"I’m really worried that if we don’t act soon, things will spiral out of control.",
			"I don’t feel safe walking home alone this late at night.",
			"What if they find out the truth and everything falls apart?",
			"I’ve got a bad feeling about this; something just doesn’t feel right.",
			"Every time I hear a noise like that, my heart races uncontrollably.",
			"I trusted you completely, and now you’ve left me feeling so hurt and betrayed.",
			"I can’t believe this is actually happening—it’s even better than I imagined!",
			"The way they acted in that situation was so horrifying, I can’t even process it.",
			"I guess I’ll just have to accept that things aren’t going to work out the way I hoped.",
			"This is such an amazing gift—I never expected something so thoughtful from you!",
			"How could you say something so cruel and completely out of line?",
			"I’m so scared that I’ll never get the chance to fix what went wrong between us.",
			"I didn’t think it would happen so soon, but I suppose we’ll figure it out.",
			"It’s great that we managed to finish early, though I’m not sure what’s next.",
			"I never thought I’d feel so heartbroken over something I didn’t see coming.",
		),
	).run()

#
# Run
#

# Main function
def main():

	parser = argparse.ArgumentParser(description="Demonstrate the TaskManager class with example applications.", add_help=False, formatter_class=functools.partial(argparse.HelpFormatter, max_help_position=36))
	parser.add_argument('--help', '-h', action='help', default=argparse.SUPPRESS, help="Show this help message and exit")
	parser.add_argument('--task', type=str, required=True, help="Which task to run (e.g. char_codes)")
	parser.add_argument('--task_prefix', type=str, metavar='PREFIX', help="Name prefix to use for task-related files (default is same as task)")
	parser.add_argument('--task_dir', type=str, metavar='DIR', help="Path to the working directory to use (will be used for automatically managed lock, task, output, state, requests and batch files, auto-created by default)")
	parser.add_argument('--chat_endpoint', type=str, metavar='ENDPOINT', default='/v1/chat/completions', help="Chat completions endpoint to use")

	parser_meta = parser.add_argument_group(title='Task metadata')  # Specifications of the task metadata to be used for new tasks (the default values are defined per-task in the corresponding task implementations)
	parser_meta.add_argument('--model', type=str, help="LLM model to use")
	parser_meta.add_argument('--max_completion_tokens', type=int, metavar='NUM', help="Maximum number of generated output tokens per request (including both reasoning and visible tokens)")
	parser_meta.add_argument('--completion_ratio', type=float, metavar='RATIO', help="How many output tokens (including both reasoning and visible tokens) to assume will be generated for each request on average, as a ratio of max_completion_tokens")
	parser_meta.add_argument('--temperature', type=float, metavar='TEMP', help="What sampling temperature to use")
	parser_meta.add_argument('--top_p', type=float, metavar='MASS', help="Nucleus sampling probability mass")
	parser_meta.add_argument('--opinions_min', type=int, metavar='NUM', help="Minimum number of successful opinions required")
	parser_meta.add_argument('--opinions_max', type=int, metavar='NUM', help="Maximum number of opinions allowed")
	parser_meta.add_argument('--confidence', type=float, metavar='RATIO', help="Opinion-based classification confidence required")

	task_manager.TaskManager.configure_argparse(parser=parser)
	gpt_requester.GPTRequester.configure_argparse(parser=parser)

	args = parser.parse_args()
	if args.task_prefix is None:
		args.task_prefix = args.task
	if args.task_dir is None:
		args.task_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks')  # Note: If library is pip-installed, then this will by default create a 'tasks' directory in the package installation location under site-packages!
	if args.wandb is None:
		args.wandb = not args.dryrun

	with gpt_requester.GPTRequester.wandb_init(config=args):

		if args.task == 'char_codes':
			demo_char_codes(cfg=args)
		elif args.task == 'utterance_emotion':
			demo_utterance_emotion(cfg=args)
		elif args.task is None:
			raise ValueError("Please specify which task to demo using --task")
		else:
			raise ValueError(f"Unrecognized task: {args.task}")

# Run main function
if __name__ == "__main__":
	log = utils.configure_logging()
	main()
# EOF
