# OpenAI GPT requester

# Imports
from __future__ import annotations
import os
import re
import copy
import enum
import http
import json
import time
import heapq
import pprint
import shutil
import getpass
import logging
import argparse
import datetime
import operator
import functools
import itertools
import contextlib
import collections
import dataclasses
from enum import auto
from typing import Optional, Union, Self, Any, ContextManager, Iterable, TextIO
import pydantic
import httpx
import httpcore  # noqa / PyCharm: Magic fix for strange if-debugging-only exception thrown when creating OpenAI client in GPTRequester __init__ (empirically, httpcore needs to be imported before breaking in debug mode at a breakpoint for the first time)
import openai
import openai._models as openai_models  # noqa
import openai.types.chat as openai_chat
import openai.lib._parsing as openai_parsing  # noqa
import openai._utils as openai_utils  # noqa
import wandb
from .logger import log
from . import tokens, utils
from .utils import NONE

# Constants
DRYRUN = '\x1b[38;5;226m[DRYRUN]\x1b[0m '
DEFAULT_ENDPOINT = '/v1/chat/completions'
FINISHED_BATCH_STATUSES = {'failed', 'completed', 'expired', 'cancelled'}
RETRYABLE_STATUS_CODES = {200, 408, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_CODES = {'batch_expired', 'batch_cancelled'}
FINISH_REASON_NORMAL = 'stop'  # Finish reason that corresponds to normal response termination
FINISH_REASONS_FAILED = {'length', 'content_filter'}  # Finish reasons that correpond to a retryable failure
FINISH_REASONS_NOCONTENT = {'content_filter', 'tool_calls', 'function_call'}  # Finish reasons for which having no content is permissible or even expected
FINISH_REASONS_ALL = set.union({FINISH_REASON_NORMAL}, FINISH_REASONS_FAILED, FINISH_REASONS_NOCONTENT)

# Logging configuration
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

#
# Common types
#

# Endpoint error
class EndpointError(Exception):
	pass

# GPT request class
@dataclasses.dataclass(frozen=True)
class GPTRequest:
	payload: dict[str, Any]                # Request payload (JSON-compatible OpenAI request after auto-parsing) => Is batched into JSONL files and uploaded to OpenAI for processing
	meta: Optional[dict[str, Any]] = None  # Optional custom metadata to associate with this request (JSON-compatible) => Is batched and stored in the state file while OpenAI is processing the payloads, and is returned alongside the response
	retry_num: int = 0                     # Retry number of the request (e.g. 2 => This request is the second retry, and thereby the third attempt at this request)

# Tokens and cost class
@dataclasses.dataclass
class TokensCost:

	input_tokens: int = 0            # An approximation of how many input tokens the request payload(s) have
	output_tokens: int = 0           # A very rough approximation of how many output tokens are expected for the request(s)
	max_output_tokens: int = 0       # The maximum number of output tokens allowed to be generated (accurate if every request explicitly specifies a maximum number of completion tokens, as requests always should)
	cost_input_direct: float = 0.0   # An approximation of the cost of the input tokens of the request payload(s) in direct mode (assuming no cached input tokens)
	cost_input_batch: float = 0.0    # An approximation of the cost of the input tokens of the request payload(s) in batch mode
	cost_output_direct: float = 0.0  # A very rough approximation of the cost of the expected number of output tokens for the request(s) in direct mode
	cost_output_batch: float = 0.0   # A very rough approximation of the cost of the expected number of output tokens for the request(s) in batch mode

	def __repr__(self) -> str:
		return f"{self.__class__.__qualname__}({', '.join(f'{field.name}={getattr(self, field.name):.3f}' if field.type == 'float' or field.type is float else f'{field.name}={getattr(self, field.name)!r}' for field in dataclasses.fields(self))})"  # noqa

	@property
	def completion_ratio(self) -> float:
		# Returns the output tokens completion ratio
		return 0.0 if self.max_output_tokens == 0 else self.output_tokens / self.max_output_tokens

	@property
	def cost_direct(self) -> float:
		# Returns the total input + output cost for direct mode
		return self.cost_input_direct + self.cost_output_direct

	@property
	def cost_batch(self) -> float:
		# Returns the total input + output cost for batch mode
		return self.cost_input_batch + self.cost_output_batch

	@property
	def cost(self) -> float:
		# Returns the total input + output cost for all modes together
		return self.cost_input_direct + self.cost_input_batch + self.cost_output_direct + self.cost_output_batch

	def equals(self, other: TokensCost, cost_tol: float = 1e-7) -> bool:
		# other = TokensCost instance to compare equality to (while avoiding exact floating point comparison)
		# cost_tol = Absolute tolerance to permit in cost comparisons
		# Returns whether the TokensCost instance is essentially equal to the other one
		return (
			self.input_tokens == other.input_tokens and
			self.output_tokens == other.output_tokens and
			self.max_output_tokens == other.max_output_tokens and
			abs(self.cost_input_direct - other.cost_input_direct) < cost_tol and
			abs(self.cost_input_batch - other.cost_input_batch) < cost_tol and
			abs(self.cost_output_direct - other.cost_output_direct) < cost_tol and
			abs(self.cost_output_batch - other.cost_output_batch) < cost_tol
		)

	def __add__(self, other: TokensCost) -> TokensCost:
		# other = TokensCost instance to add to the current one
		# Returns the sum as a new TokensCost instance
		if not isinstance(other, TokensCost):
			return NotImplemented
		return TokensCost(
			input_tokens=self.input_tokens + other.input_tokens,
			output_tokens=self.output_tokens + other.output_tokens,
			max_output_tokens=self.max_output_tokens + other.max_output_tokens,
			cost_input_direct=self.cost_input_direct + other.cost_input_direct,
			cost_input_batch=self.cost_input_batch + other.cost_input_batch,
			cost_output_direct=self.cost_output_direct + other.cost_output_direct,
			cost_output_batch=self.cost_output_batch + other.cost_output_batch,
		)

	def __iadd__(self, other: TokensCost) -> Self:
		# other = TokensCost instance to add in-place to the current one
		# Returns the updated (self) TokensCost instance
		if not isinstance(other, TokensCost):
			return NotImplemented
		self.input_tokens += other.input_tokens
		self.output_tokens += other.output_tokens
		self.max_output_tokens += other.max_output_tokens
		self.cost_input_direct += other.cost_input_direct
		self.cost_input_batch += other.cost_input_batch
		self.cost_output_direct += other.cost_output_direct
		self.cost_output_batch += other.cost_output_batch
		return self

	def add(self, *others: Union[TokensCost, Iterable[TokensCost]]) -> Self:
		# others = TokensCost instances or iterables thereof to add in-place to the current one
		# Returns the updated (self) TokensCost instance
		for other in others:
			if isinstance(other, TokensCost):
				self.__iadd__(other)
			elif isinstance(other, Iterable):
				for oth in other:
					self.__iadd__(oth)
			else:
				raise TypeError(f"Unexpected type: {utils.get_class_str(other)}")
		return self

	def is_valid(self) -> bool:
		# Returns whether all fields are non-negative
		return self.input_tokens >= 0 and self.output_tokens >= 0 and self.max_output_tokens >= 0 and self.cost_input_direct >= 0 and self.cost_input_batch >= 0 and self.cost_output_direct >= 0 and self.cost_output_batch >= 0

#
# State file
#

# Queue state class
@dataclasses.dataclass
class QueueState:
	request_id_meta: dict[int, Optional[dict[str, Any]]] = dataclasses.field(default_factory=dict)  # A map of request ID to custom metadata for all requests in the request queue (NOT including the request pool, ordered by ascending request ID)

# Request information class
@dataclasses.dataclass(frozen=True)
class RequestInfo:
	meta: Optional[dict[str, Any]]        # Optional custom metadata that is associated with this request (JSON-compatible)
	retry_num: int                        # Retry number of the request (e.g. 2 => This request is the second retry, and thereby the third attempt at this request)
	tokens_cost: TokensCost               # The approximate token count and cost of the request payload
	parse_info: Optional[dict[str, Any]]  # Data required for or associated with auto-parsing of the request

# Remote batch state class
@dataclasses.dataclass
class RemoteBatchState:
	file: openai.types.FileObject  # Uploaded input JSONL file object
	batch: openai.types.Batch      # Batch status as per the last time the server was checked (this is not updated anymore once the batch status is for the first time in one of the finished states)
	finished: bool                 # Whether the batch is known to have finished on the remote server by now (status as per the last time the server was checked, updated whenever status is updated)

# Request metrics class
@dataclasses.dataclass
class RequestMetrics:

	num_requests: int = 0                                                                                        # Total number of completed requests
	local_jsonl_size: int = 0                                                                                    # Total number of completed request bytes (as per how many bytes the requests are in the local JSONL batch input files)
	models: collections.Counter[str] = dataclasses.field(default_factory=collections.Counter)                    # Models used for completed requests
	system_fingerprints: collections.Counter[str] = dataclasses.field(default_factory=collections.Counter)       # System fingerprints seen for completed requests
	usage: dict[str, Union[int, float, dict[str, Union[int, float]]]] = dataclasses.field(default_factory=dict)  # Total true usage associated with completed requests (i.e. token usage)
	max_output_tokens: int = 0                                                                                   # Maximum allowed number of generated output tokens associated with completed requests (each request specifies in its request payload the maximum number of completion tokens)
	max_seen_output_tokens: int = 0                                                                              # Maximum seen true number of output tokens for a request
	true_cost: float = 0.0                                                                                       # Total true cost associated with completed requests

	@property
	def max_output_tokens_avg(self) -> float:
		return 0.0 if self.num_requests == 0 else self.max_output_tokens / self.num_requests

	@property
	def output_tokens_avg(self) -> float:
		return 0.0 if self.num_requests == 0 else self.usage.get('completion_tokens', 0) / self.num_requests

	@property
	def completion_ratio(self) -> float:
		# Returns the true output tokens completion ratio
		return 0.0 if self.max_output_tokens == 0 else self.usage.get('completion_tokens', 0) / self.max_output_tokens

	def __add__(self, other: RequestMetrics) -> RequestMetrics:
		# other = Another RequestMetrics instance to combine with this one
		# Returns a new RequestMetrics instance with combined metrics
		if not isinstance(other, RequestMetrics):
			return NotImplemented
		return RequestMetrics(
			num_requests=self.num_requests + other.num_requests,
			local_jsonl_size=self.local_jsonl_size + other.local_jsonl_size,
			models=self.models + other.models,
			system_fingerprints=self.system_fingerprints + other.system_fingerprints,
			usage=RequestMetrics.usage_iadd(self_usage=copy.deepcopy(self.usage), other_usage=other.usage),
			max_output_tokens=self.max_output_tokens + other.max_output_tokens,
			max_seen_output_tokens=max(self.max_seen_output_tokens, other.max_seen_output_tokens),
			true_cost=self.true_cost + other.true_cost,
		)

	def __iadd__(self, other: RequestMetrics) -> Self:
		# other = Another RequestMetrics instance to merge into this one
		# Returns the updated (self) RequestMetrics instance
		if not isinstance(other, RequestMetrics):
			return NotImplemented
		self.num_requests += other.num_requests
		self.local_jsonl_size += other.local_jsonl_size
		self.models.update(other.models)
		self.system_fingerprints.update(other.system_fingerprints)
		RequestMetrics.usage_iadd(self_usage=self.usage, other_usage=other.usage)
		self.max_output_tokens += other.max_output_tokens
		self.max_seen_output_tokens = max(self.max_seen_output_tokens, other.max_seen_output_tokens)
		self.true_cost += other.true_cost
		return self

	def add_metrics(self, *others: Union[RequestMetrics, Iterable[RequestMetrics]]) -> Self:
		# others = RequestMetrics instances or iterables thereof to add in-place to the current one
		# Returns the updated (self) RequestMetrics instance
		for other in others:
			if isinstance(other, RequestMetrics):
				self.__iadd__(other)
			elif isinstance(other, Iterable):
				for oth in other:
					self.__iadd__(oth)
			else:
				raise TypeError(f"Unexpected type: {utils.get_class_str(other)}")
		return self

	@staticmethod
	def usage_iadd(self_usage: dict[str, Union[int, float, dict[str, Union[int, float]]]], other_usage: dict[str, Union[int, float, dict[str, Union[int, float]]]]) -> dict[str, Union[int, float, dict[str, Union[int, float]]]]:
		# self_usage = The usage to update
		# other_usage = The usage to add/merge into self_usage
		# Returns the updated self_usage
		for key, value in other_usage.items():
			if key in self_usage:
				self_value = self_usage[key]
				if isinstance(self_value, dict) and isinstance(value, dict):
					RequestMetrics.usage_iadd(self_usage=self_value, other_usage=value)
				elif isinstance(self_value, (int, float)) and isinstance(value, (int, float)):
					self_usage[key] += value
				else:
					raise TypeError(f"Type mismatch while trying to add usages: {utils.get_class_str(self_value)} vs {utils.get_class_str(value)}")
			else:
				self_usage[key] = value if isinstance(value, (int, float)) else copy.deepcopy(value)
		return self_usage

# API metrics class
@dataclasses.dataclass
class APIMetrics:
	num_api_calls: int = 0                                                         # Number of API calls that were associated with the completed requests
	succeeded: RequestMetrics = dataclasses.field(default_factory=RequestMetrics)  # Metrics associated with succeeded requests
	failed: RequestMetrics = dataclasses.field(default_factory=RequestMetrics)     # Metrics associated with failed requests
	total: RequestMetrics = dataclasses.field(default_factory=RequestMetrics)      # Total metrics of completed requests for the API

# Metrics class
@dataclasses.dataclass
class Metrics:
	direct: APIMetrics = dataclasses.field(default_factory=APIMetrics)  # Metrics associated with completed direct API requests
	batch: APIMetrics = dataclasses.field(default_factory=APIMetrics)   # Metrics associated with completed batch API requests
	all: APIMetrics = dataclasses.field(default_factory=APIMetrics)     # Total metrics of completed requests across all APIs

# Batch state class
@dataclasses.dataclass(order=True)
class BatchState:
	id: int = 0                                                                     # Unique monotonic ID corresponding to the batch
	local_jsonl: str = ''                                                           # Path of the generated local JSONL batch input file (requests are ordered by ascending request ID)
	local_jsonl_size: int = 0                                                       # Number of bytes in the local JSONL batch input file
	num_requests: int = 0                                                           # Number of requests in the batch
	request_info: dict[int, RequestInfo] = dataclasses.field(default_factory=dict)  # The extra information pertaining to each request
	tokens_cost: TokensCost = dataclasses.field(default_factory=TokensCost)         # The total approximate token count and cost across all requests in the batch
	full_batch: bool = False                                                        # Whether this is a full batch, i.e. some threshold was reached that triggered this batch to be formed, as opposed to the batch being forced
	reasons: list[str] = dataclasses.field(default_factory=list)                    # String reasons of what triggered the formation of the batch
	remote: Optional[RemoteBatchState] = None                                       # Remote state of the batch once it has been pushed
	true_tokens_cost: Optional[TokensCost] = None                                   # True total token count and cost across all responded requests in the batch (only available once batch has finished and been processed)
	metrics: Optional[APIMetrics] = None                                            # True metrics associated with the batch (only available once batch has finished and been processed)

# Push stats class
@dataclasses.dataclass
class PushStats:
	total_requests: int = 0                                                        # Total number of requests pushed
	total_tokens_cost: TokensCost = dataclasses.field(default_factory=TokensCost)  # Total token count and cost pushed (cost is added only in the fields corresponding to the API that was used in the push)

# State class
@dataclasses.dataclass
class State:

	version: int = 1                                                            # Data format version number
	endpoint: str = ''                                                          # API endpoint to use (empty string is invalid)
	max_request_id: int = 0                                                     # Maximum request ID used thus far (0 = No request ID used so far)
	max_batch_id: int = 0                                                       # Maximum batch ID used thus far (0 = No batch ID used so far)
	max_direct_batch_id: int = 0                                                # Maximum direct batch ID used thus far (0 = No direct batch ID used so far)
	queue: QueueState = dataclasses.field(default_factory=QueueState)           # State of the request queue
	batches: list[BatchState] = dataclasses.field(default_factory=list)         # States of all batches that are currently pending or in progress (in ascending batch ID order)
	batch_history: list[BatchState] = dataclasses.field(default_factory=list)   # History of the final states of the completed batches (in ascending batch ID order, with some overdetailed per-request information removed for space reasons)
	direct_history: list[BatchState] = dataclasses.field(default_factory=list)  # History of the final states of the completed direct batches (in ascending batch ID order, with some overdetailed per-request information removed for space reasons)
	task_push_stats: PushStats = dataclasses.field(default_factory=PushStats)   # Task push statistics
	metrics: Metrics = dataclasses.field(default_factory=Metrics)               # Completed request metrics
	num_pass_failures: int = 0                                                  # Current number of consecutive batch pass failures

	def summary(self) -> dict[str, Any]:
		# Returns a flat JSON-compatible summary of the data contained in the class

		unpushed_tokens_cost = TokensCost().add(batch.tokens_cost for batch in self.batches if batch.remote is None)
		remote_tokens_cost = TokensCost().add(batch.tokens_cost for batch in self.batches if batch.remote is not None)
		ongoing_tokens_cost = unpushed_tokens_cost + remote_tokens_cost

		usage = {}
		for key, value in self.metrics.all.total.usage.items():
			if isinstance(value, dict):
				key = key.removesuffix('_details')
				key_is_tokens = key.endswith('_tokens')
				for subkey, subvalue in value.items():
					if isinstance(subvalue, (int, float)):
						usage_key = f'Usage/{key}_{subkey}'
						if usage_key.endswith('_tokens'):
							if key_is_tokens:
								usage_key = usage_key[:-7]
							usage_key = usage_key.replace('_tokens', '_ktok')
							subvalue /= 1000
						usage[usage_key] = subvalue
			elif isinstance(value, (int, float)):
				usage_key = f'Usage/{key}'
				if usage_key.endswith('_tokens'):
					usage_key = usage_key.replace('_tokens', '_ktok')
					value /= 1000
				usage[usage_key] = value

		return {
			'Metrics/max_request_id': self.max_request_id,
			'Batch/max_batch_id': self.max_batch_id,
			'Direct/max_direct_id': self.max_direct_batch_id,
			'Batch/num_total': len(self.batches) + len(self.batch_history),
			'Batch/num_completed': len(self.batch_history),
			'Batch/num_ongoing': len(self.batches),
			'Batch/num_unpushed': sum(batch.remote is None for batch in self.batches),
			'Batch/num_remote': sum(batch.remote is not None for batch in self.batches),
			'Batch/ongoing_requests': sum(batch.num_requests for batch in self.batches),
			'Batch/ongoing_json_size_mb': sum(batch.local_jsonl_size for batch in self.batches) / 1048576,
			'Batch/ongoing_input_ktok': ongoing_tokens_cost.input_tokens / 1000,
			'Batch/ongoing_pred_cost': ongoing_tokens_cost.cost_batch,
			'Batch/unpushed_requests': sum(batch.num_requests for batch in self.batches if batch.remote is None),
			'Batch/unpushed_json_size_mb': sum(batch.local_jsonl_size for batch in self.batches if batch.remote is None) / 1048576,
			'Batch/unpushed_input_ktok': unpushed_tokens_cost.input_tokens / 1000,
			'Batch/unpushed_pred_cost': unpushed_tokens_cost.cost_batch,
			'Batch/remote_requests': sum(batch.num_requests for batch in self.batches if batch.remote is not None),
			'Batch/remote_requests_completed': sum(batch.remote.batch.request_counts.completed for batch in self.batches if batch.remote is not None and batch.remote.batch.request_counts is not None),
			'Batch/remote_requests_failed': sum(batch.remote.batch.request_counts.failed for batch in self.batches if batch.remote is not None and batch.remote.batch.request_counts is not None),
			'Batch/remote_requests_running': sum(batch.remote.batch.request_counts.total for batch in self.batches if batch.remote is not None and batch.remote.batch.request_counts is not None),
			'Batch/remote_json_size_mb': sum(batch.local_jsonl_size for batch in self.batches if batch.remote is not None) / 1048576,
			'Batch/remote_input_ktok': remote_tokens_cost.input_tokens / 1000,
			'Batch/remote_pred_cost': remote_tokens_cost.cost_batch,
			'Batch/completed_requests': sum(batch.num_requests for batch in self.batch_history),
			'Batch/completed_cost': sum(batch.true_tokens_cost.cost_batch for batch in self.batch_history),
			'Direct/num_completed': len(self.direct_history),
			'Direct/completed_requests': sum(batch.num_requests for batch in self.direct_history),
			'Direct/completed_cost': sum(batch.true_tokens_cost.cost_direct for batch in self.direct_history),
			'Pushed/task_requests': self.task_push_stats.total_requests,
			'Pushed/task_input_ktok': self.task_push_stats.total_tokens_cost.input_tokens / 1000,
			'Pushed/task_pred_cost': self.task_push_stats.total_tokens_cost.cost,
			'Metrics/num_api_calls': self.metrics.all.num_api_calls,
			'Metrics/num_requests': self.metrics.all.total.num_requests,
			'Metrics/json_size_mb': self.metrics.all.total.local_jsonl_size / 1048576,
			'Metrics/num_models': len(self.metrics.all.total.models),
			'Metrics/num_fingerprints': len(self.metrics.all.total.system_fingerprints),
			**usage,
			'Metrics/completion_ratio': self.metrics.all.total.completion_ratio,
			'Metrics/max_completion_tokens_avg': self.metrics.all.total.max_output_tokens_avg,
			'Metrics/num_completion_tokens_avg': self.metrics.all.total.output_tokens_avg,
			'Metrics/longest_completion_tokens': self.metrics.all.total.max_seen_output_tokens,
			'Metrics/cost': self.metrics.all.total.true_cost,
			'Batch/consec_pass_failures': self.num_pass_failures,
		}

# State file class
class StateFile:

	state: Optional[State]

	def __init__(self, path: str, endpoint: str, dryrun: bool, W: Optional[utils.WandbRun] = None):
		# path = Path to the JSON state file to load/save/manage (nominally *.json extension)
		# endpoint = API endpoint to use
		# dryrun = Whether to prevent any saving of state (dry run mode)
		# W = Wandb run holder
		self.path = os.path.abspath(path)
		self.name = os.path.basename(self.path)
		log.info(f"GPT requester state file: {self.path}")
		self.endpoint = endpoint
		if not self.endpoint:
			raise ValueError("State API endpoint must be non-empty")
		self.dryrun = dryrun
		self.W = W if W is not None else utils.WandbRun()
		self._enter_stack = contextlib.ExitStack()
		self.state = None

	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicLogRevertStack(W=self.W) as rstack:
				enter_stack.callback(self.unload)
				try:
					self.load(rstack=rstack)
				except FileNotFoundError:
					self.create(rstack=rstack)
				assert self.state is not None
				if not self.state.endpoint:
					raise ValueError("State API endpoint must be non-empty")
				if self.state.endpoint != self.endpoint:
					raise ValueError(f"State API endpoint mismatch: {self.state.endpoint} vs {self.endpoint}")
			self._enter_stack = enter_stack.pop_all()
		assert self._enter_stack is not enter_stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for the safe reversible creation of the state file
		rstack.callback(setattr, self, 'state', self.state)
		self.state = State(endpoint=self.endpoint)
		self.save(rstack=rstack)

	def load(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for the safe reversible loading of the state file
		rstack.callback(setattr, self, 'state', self.state)
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.state = utils.dataclass_from_json(cls=State, json_data=file)
		log.info(f"Loaded GPT requester state file with {len(self.state.queue.request_id_meta)} queued requests and {len(self.state.batches)} batches ({utils.format_size_iec(file_size)})")
		self.log_summary(rstack=rstack)

	def save(self, rstack: utils.RevertStack, show_log: bool = True):
		# rstack = RevertStack to use for the safe reversible saving of the state file
		# show_log = Whether to log that the state file was saved
		if self.dryrun:
			if show_log:
				log.warning(f"{DRYRUN}Did not save GPT requester state file with {len(self.state.queue.request_id_meta)} queued requests and {len(self.state.batches)} batches")
		else:
			with utils.SafeOpenForWrite(path=self.path, rstack=rstack) as file:
				utils.json_from_dataclass(obj=self.state, file=file)
				file_size = utils.get_file_size(file)
			if show_log:
				log.info(f"Saved GPT requester state file with {len(self.state.queue.request_id_meta)} queued requests and {len(self.state.batches)} batches ({utils.format_size_iec(file_size)})")
		self.log_summary(rstack=rstack)

	def unload(self):
		self.state = None

	def log_summary(self, rstack: utils.RevertStack):
		# rstack = Possible LogRevertStack (subclass of RevertStack) to use for logging
		if isinstance(rstack, utils.LogRevertStack):
			rstack.log(self.state.summary())

#
# Queue file
#

# GPT request item class (each line of the JSONL queue file corresponds to one of these items)
@dataclasses.dataclass(frozen=True, order=True)
class GPTRequestItem:
	id: int                               # Unique monotonic ID corresponding to the GPT request
	req: GPTRequest                       # The actual GPT request
	parse_info: Optional[dict[str, Any]]  # Data required for or associated with auto-parsing of the request

# GPT request information class
@dataclasses.dataclass(frozen=True)
class GPTRequestInfo:
	json: str                # Full batch-style request in compact JSON string form (includes the payload, always ends with a single trailing newline)
	json_size: int           # Number of bytes in the compact JSON string form when encoded with UTF-8
	tokens_cost: TokensCost  # The approximate token count and cost of the request payload

# Cached GPT request class
@dataclasses.dataclass(frozen=True, order=True)
class CachedGPTRequest:

	item: GPTRequestItem  # GPT request item (backed by the queue file)
	info: GPTRequestInfo  # GPT request information (automatically calculated from the GPT request item, and cached in memory)

	@staticmethod
	def from_item(item: GPTRequestItem, endpoint: str, token_estimator: tokens.TokenEstimator, token_coster: tokens.TokenCoster) -> CachedGPTRequest:
		# item = GPT request item to calculate information for and wrap as a cached GPT request
		# endpoint = API endpoint to use
		# token_estimator = Token estimator to use to estimate the number of tokens in the request
		# token_coster = Token coster to use to estimate the cost of the request
		# Returns the created cached GPT request
		full_request = dict(custom_id=f'id-{item.id}', method='POST', url=endpoint, body=item.req.payload)
		compact_json = json.dumps(full_request, ensure_ascii=False, indent=None) + '\n'
		input_tokens = token_estimator.payload_input_tokens(payload=item.req.payload, endpoint=endpoint).total
		output_tokens, max_output_tokens = token_estimator.payload_output_tokens(payload=item.req.payload, endpoint=endpoint)
		return CachedGPTRequest(
			item=item,
			info=GPTRequestInfo(
				json=compact_json,
				json_size=len(compact_json.encode('utf-8')),
				tokens_cost=TokensCost(
					input_tokens=input_tokens,
					output_tokens=output_tokens,
					max_output_tokens=max_output_tokens,
					cost_input_direct=token_coster.input_cost(direct=input_tokens),
					cost_input_batch=token_coster.input_cost(batch=input_tokens),
					cost_output_direct=token_coster.output_cost(direct=output_tokens),
					cost_output_batch=token_coster.output_cost(batch=output_tokens),
				),
			),
		)

	def get_request_info(self) -> RequestInfo:
		# Returns a newly created (shallow) RequestInfo based on the CachedGPTRequest
		return RequestInfo(meta=self.item.req.meta, retry_num=self.item.req.retry_num, tokens_cost=self.info.tokens_cost, parse_info=self.item.parse_info)

# Pool/queue class
@dataclasses.dataclass(frozen=True)
class PoolQueue:

	pool: list[CachedGPTRequest] = dataclasses.field(default_factory=list)             # Request pool: Requests added to the GPT requester are first added to the request pool, which is purely in memory and not backed by a file (the request pool is 'lost' if a KeyboardInterrupt or crash occurs)
	queue: list[Optional[CachedGPTRequest]] = dataclasses.field(default_factory=list)  # Request queue: When the GPT requester is made to commit, all requests in the request pool are moved to the request queue, which is backed by the queue file and thus persistent

	@property
	def pool_len(self) -> int:
		# Returns the number of requests in the request pool
		return len(self.pool)

	@property
	def queue_len(self) -> int:
		# Returns the number of requests in the request queue
		return sum(cached_req is not None for cached_req in self.queue)

	def queue_request_id_meta(self) -> dict[int, Optional[dict[str, Any]]]:
		# Returns an ID to metadata map for the current state of the request queue
		request_id_meta = {cached_req.item.id: cached_req.item.req.meta for cached_req in self.queue if cached_req is not None}
		if len(request_id_meta) != self.queue_len:
			raise ValueError("Request queue contains duplicate request IDs")
		return request_id_meta

	def summary(self) -> dict[str, Any]:
		# Returns a flat JSON-compatible summary of the data contained in the class
		total_json_size = sum(cached_req.info.json_size for cached_req in self.pool) + sum(cached_req.info.json_size for cached_req in self.queue if cached_req is not None)
		total_tokens_cost = TokensCost().add((cached_req.info.tokens_cost for cached_req in self.pool), (cached_req.info.tokens_cost for cached_req in self.queue if cached_req is not None))
		return {
			'Queue/pool_len': self.pool_len,
			'Queue/queue_len': self.queue_len,
			'Queue/num_requests': self.pool_len + self.queue_len,
			'Queue/json_size_mb': total_json_size / 1048576,
			'Queue/input_ktok': total_tokens_cost.input_tokens / 1000,
			'Queue/max_output_ktok': total_tokens_cost.max_output_tokens / 1000,
			'Queue/pred_output_ktok': total_tokens_cost.output_tokens / 1000,
			'Queue/pred_completion_ratio': total_tokens_cost.completion_ratio,
			'Queue/pred_cost_direct': total_tokens_cost.cost_direct,
			'Queue/pred_cost_batch': total_tokens_cost.cost_batch,
		}

# Queue file class
class QueueFile:

	pool_queue: Optional[PoolQueue]

	def __init__(self, path: str, endpoint: str, token_estimator: tokens.TokenEstimator, token_coster: tokens.TokenCoster, dryrun: bool, W: Optional[utils.WandbRun] = None):
		# path = Path to the JSONL queue file to load/save/manage (nominally *.jsonl extension)
		# endpoint = API endpoint to use
		# token_estimator = Token estimator instance to use
		# token_coster = Token coster instance to use
		# dryrun = Whether to prevent any saving of the queue file (dry run mode)
		# W = Wandb run holder
		self.path = os.path.abspath(path)
		self.name = os.path.basename(self.path)
		log.info(f"GPT requester queue file: {self.path}")
		self.endpoint = endpoint
		self.token_estimator = token_estimator
		self.token_coster = token_coster
		self.dryrun = dryrun
		self.W = W if W is not None else utils.WandbRun()
		self._enter_stack = contextlib.ExitStack()
		self.pool_queue = None

	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			with utils.AtomicLogRevertStack(W=self.W) as rstack:
				enter_stack.callback(self.unload)
				try:
					self.load(rstack=rstack)
				except FileNotFoundError:
					self.create(rstack=rstack)
				assert self.pool_queue is not None
			self._enter_stack = enter_stack.pop_all()
		assert self._enter_stack is not enter_stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	def create(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible creation of the queue file
		rstack.callback(setattr, self, 'pool_queue', self.pool_queue)
		self.pool_queue = PoolQueue()
		self.save(rstack=rstack)

	def load(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible loading of the queue file
		rstack.callback(setattr, self, 'pool_queue', self.pool_queue)
		with open(self.path, 'r', encoding='utf-8') as file:
			file_size = utils.get_file_size(file)
			self.pool_queue = PoolQueue(pool=[], queue=[CachedGPTRequest.from_item(
				item=utils.dataclass_from_json(cls=GPTRequestItem, json_data=line),
				endpoint=self.endpoint,
				token_estimator=self.token_estimator,
				token_coster=self.token_coster,
			) for line in file])
		log.info(f"Loaded GPT requester queue file with {self.pool_queue.queue_len} requests ({utils.format_size_iec(file_size)})")
		self.log_summary(rstack=rstack)

	def save(self, rstack: utils.RevertStack):
		# rstack = RevertStack to use for safe reversible saving of the queue file
		if self.dryrun:
			log.warning(f"{DRYRUN}Did not save GPT requester queue file with {self.pool_queue.queue_len} requests")
		else:
			with utils.SafeOpenForWrite(path=self.path, rstack=rstack) as file:
				for cached_req in self.pool_queue.queue:
					if cached_req is not None:
						utils.json_from_dataclass(obj=cached_req.item, file=file, indent=None)
						file.write('\n')
				file_size = utils.get_file_size(file)
			log.info(f"Saved GPT requester queue file with {self.pool_queue.queue_len} requests ({utils.format_size_iec(file_size)})")
		self.log_summary(rstack=rstack)

	def unload(self):
		self.pool_queue = None

	def log_summary(self, rstack: utils.RevertStack):
		# rstack = Possible LogRevertStack (subclass of RevertStack) to use for logging
		if isinstance(rstack, utils.LogRevertStack):
			rstack.log(self.pool_queue.summary())

#
# Responses
#

# Response information class
@dataclasses.dataclass(frozen=True)
class ResponseInfo:
	payload: Union[openai_chat.ChatCompletion, openai_chat.ParsedChatCompletion[object]]  # Response payload (type depends on endpoint and auto-parsing)
	model: str                                                                            # Model that was used for the response
	system_fingerprint: str                                                               # System fingerprint of the system that generated the response (empty string if none provided by remote)
	usage: dict[str, Union[int, float, dict[str, Union[int, float]]]]                     # Response usage (part of the raw payload) as a nested dict

# Error information class
@dataclasses.dataclass(frozen=True)
class ErrorInfo:
	fatal: bool                 # Whether the error is fatal (True, should raise an exception) or retryable (False)
	type: str                   # Type of the error
	msg: str                    # String message summarizing the error
	subtype: str = ''           # Sub-type of the error (may be empty string if there is no division by sub-type for this error type)
	data: Optional[Any] = None  # Optional data associated with the error (the type of the data depends on the type of the error)

# Result information class
@dataclasses.dataclass
class ResultInfo:
	req_id: int                        # Request ID
	req_jsonl_size: int                # Request JSONL size (bytes)
	req_payload: dict[str, Any]        # Request payload (JSON-loaded)
	req_info: RequestInfo              # Request information
	resp_info: Optional[ResponseInfo]  # Response information (if available / both resp_info and err_info may be None if dry run is active)
	err_info: Optional[ErrorInfo]      # Error information (if errored / both resp_info and err_info may be None if dry run is active)
	warn_infos: list[ErrorInfo]        # Warning information (if warnings occurred)
	retry_counts: bool                 # Whether a retry of this request counts towards the maximum retry count (e.g. by default a retry due to batch cancellation or expiration does not count as a retry)
	retry: bool                        # Whether the request should be retried (as something retryable went wrong)

# Result statistics class
@dataclasses.dataclass(frozen=True)
class ResultStats:
	num_requests: int      # Number of requests
	num_success: int       # Number of results with no warnings or error
	num_empty: int         # Number of results with neither a response nor an error
	num_warned: int        # Number of results with warnings but no error
	num_errored: int       # Number of results with an error
	num_retryable: int     # Number of results with a retryable error
	num_cancelled: int     # Number of results with a retryable error due to the batch being cancelled
	num_expired: int       # Number of results with a retryable error due to the batch being expired
	num_fatal: int         # Number of results with a fatal error
	num_missing: int       # Number of requests for which no response/result was received
	num_pass: int          # Number of results considered as 'passed' (i.e. not indicative of problems = successful or expired)
	pass_ratio: float      # Ratio of passed results to the number of requests
	num_final: int         # Number of final results (results that are NOT flagged as needing to be retried)
	num_final_err: int     # Number of final results with an error
	num_final_warn: int    # Number of final results with warnings but no error
	num_final_none: int    # Number of final results with no warnings or error
	num_retry: int         # Number of retry results (results that are flagged as needing to be retried)
	num_retry_err: int     # Number of retry results with an error
	num_retry_warn: int    # Number of retry results with warnings but no error
	num_retry_none: int    # Number of retry results with no warnings or error
	num_retry_counts: int  # Number of retry results for which the retry counts towards the retry number

# Batch result class
@dataclasses.dataclass
class BatchResult:
	batch: BatchState                      # Final batch state including final remote state (prior to any removal of overdetailed per-request information for the purpose of saving the batch to the history)
	duration: int                          # Duration in seconds it took for the batch to finalize on the remote (complete, cancel, ...)
	duration_hmin: str                     # Duration formatted as a 'XhYm' string (see utils.format_duration_hmin)
	errors: list[openai.types.BatchError]  # List of any encountered batch errors
	info: dict[int, ResultInfo]            # Result information for each request ID in the batch (in ascending request ID order, guaranteed to be for exactly all requests in the batch)
	stats: ResultStats                     # Result statistics

#
# GPT requester
#

# Direct verbose mode enumeration
class DirectVerboseMode(utils.OrderedEnumLU):
	NEVER = auto()
	ERROR = auto()
	WARN = auto()
	ALWAYS = auto()

# Verbose pretty printer
class VerbosePrettyPrinter(pprint.PrettyPrinter):

	def __init__(self, depth_width: int = 20, depth_extra: int = 40, depth_min: int = 3):
		self.depth_width = max(depth_width, 3)
		self.depth_extra = max(depth_extra, 5)
		self.depth_min = max(depth_min, 1)
		self.width = max(shutil.get_terminal_size().columns, 80)
		self.depth = max((self.width - self.depth_extra) // self.depth_width, self.depth_min)
		super().__init__(indent=2, width=self.width, depth=self.depth, compact=True, sort_dicts=False)

	# noinspection PyUnresolvedReferences, PyProtectedMember, PyTypeChecker
	def _format(self, obj: object, stream: TextIO, indent: int, allowance: int, context: dict[int, int], level: int):
		if level == 0 and dataclasses._is_dataclass_instance(obj):
			obj = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
		elif isinstance(obj, pydantic.BaseModel):
			obj = {k: getattr(obj, k) for k in obj.model_fields}
		elif isinstance(obj, str) and len(obj) > self.depth_extra - 2 and re.fullmatch(r"data:image/[a-zA-Z]{3,4};base64,[a-zA-Z0-9+/]{2,}={0,2}", obj):
			obj = f"{obj[:(self.depth_extra - 5)]}..."
		super()._format(object=obj, stream=stream, indent=indent, allowance=allowance, context=context, level=level)
		if level == 0 and isinstance(obj, dict) and stream.seekable() and stream.tell() > 0:
			stream.truncate(stream.tell() - 1)
			stream.seek(0)
			stream.write(' ')

# GPT requester class
@utils.init_kwargs
class GPTRequester:

	assumed_completion_ratio: Optional[float]

	# Construct a GPT requester to make use of the OpenAI Batch API to process GPT requests
	def __init__(
		self,
		working_dir: str,                                         # Path to the GPT working directory to use (will be used for automatically managed lock, state, requests and batch files)
		name_prefix: str,                                         # Name prefix to use for all files created in the GPT working directory (e.g. 'my_gpt_requester')
		*,                                                        # Keyword arguments only beyond here

		dryrun: bool = False,                                     # Whether to perform a dry run (e.g. no OpenAI API calls, no pushing batches to remote, no processing finished batches, no writing to disk, ...)
		only_process: bool = False,                               # Whether to solely process existing unfinished remote batches and not generate/commit/push anything new
		process_failed_batches: int = 0,                          # Whether to force the processing of any failed batches, thereby finalizing them and clearing them from the remote and pipeline (-1 = Process all failed batches, 0 = Do not process any failed batches, >0 = Process at most N failed batches in this session)
		retry_fatal_requests: bool = False,                       # Whether to retry fatal requests from failed batches that are processed, or otherwise skip and fail them
		wipe_requests: bool = False,                              # CAUTION: Wipe and forget all ongoing requests and request batches without processing them (cancels/deletes batches from remote, does not affect any already finished/processed requests, consider running the requester with only_process=True prior to wiping)
		wipe_task: bool = False,                                  # CAUTION: Wipe and forget all progress and existing results on the task, and start completely afresh from scratch (implies wipe_requests first, also triggers task manager actions if requester is being managed by one)

		openai_api_key: Optional[str] = None,                     # OpenAI API key (see openai.OpenAI, ends up in request headers)
		openai_organization: Optional[str] = None,                # OpenAI organization (see openai.OpenAI, ends up in request headers)
		openai_project: Optional[str] = None,                     # OpenAI project (see openai.OpenAI, ends up in request headers)
		client_base_url: Union[str, httpx.URL, None] = None,      # Base URL to use for the OpenAI API client (see openai.OpenAI, servers other than OpenAI's servers can be configured to expose an OpenAI API with suitable endpoints)
		client_kwargs: Optional[dict[str, Any]] = None,           # Additional kwargs to use for the OpenAI API client (see openai.OpenAI)
		client: Optional[openai.OpenAI] = None,                   # OpenAI API client instance to use, if given, otherwise one is created internally (Note: If an explicit client instance is given, the preceding client_* and openai_* arguments are ignored)
		endpoint: Optional[str] = None,                           # Endpoint to use for all GPT requests (None => OPENAI_ENDPOINT environment variable with the DEFAULT_ENDPOINT constant as a fallback)

		wandb: Optional[bool] = None,                             # noqa / Whether wandb logging is enabled
		wandb_dir: Optional[str] = None,                          # Wandb directory (must already exist, see also the WANDB_DIR environment variable)
		wandb_entity: Optional[str] = None,                       # Wandb entity
		wandb_project: Optional[str] = 'gpt_batch_api',           # Wandb project name
		wandb_run_id: Optional[str] = None,                       # Wandb run ID to create or resume (auto-generated run IDs are typically lowercase alphanumeric strings of length 8)
		wandb_name: Optional[str] = None,                         # Wandb run name
		wandb_group: Optional[str] = None,                        # Wandb group (defaults to the class name of the task manager class)

		autocreate_working_dir: bool = True,                      # Whether to automatically create the GPT working directory if it does not exist (parent directory must already exist)
		lock_timeout: Optional[float] = None,                     # Timeout (if any) to use when attempting to lock exclusive access to the files in the GPT working directory corresponding to the given name prefix (see utils.LockFile)
		lock_poll_interval: Optional[float] = None,               # Lock file polling interval (see utils.LockFile)
		lock_status_interval: Optional[float] = None,             # Lock file status update interval (see utils.LockFile)
		token_estimator_warn: str = 'once',                       # Warning mode to use for internal token estimator (see tokens.TokenEstimator)
		remote_update_interval: float = 10.0,                     # Interval in multiples of which to update remote batch states when waiting for remote batches to finish (seconds)
		wait_save_interval: float = 900.0,                        # Minimum time interval in seconds between successive state saves to disk while waiting for batches to complete

		direct_verbose: Union[str, DirectVerboseMode] = 'never',  # Whether to show verbose requests/responses when using the direct API (options: never, error, warn (i.e. if a warning or error occurs), always)
		show_warnings: int = 50,                                  # How many warnings to show/log per batch per warning type when processing responses
		show_errors: int = 25,                                    # How many errors to show/log per batch per error type when processing responses
		max_retries: int = 2,                                     # Maximum number of retries to allow by default for a request (e.g. 2 retries => 3 attempts allowed, retries due to batch cancellation or expiration do not count towards the retry count)
		auto_parse: bool = True,                                  # Whether to perform auto-parsing to help validate and Python-ify API requests and responses

		cost_input_direct_mtoken: float = 2.50,                   # The cost per million direct input tokens (1M tokens ~ 750K words)
		cost_input_cached_mtoken: float = 1.25,                   # The cost per million cached input tokens
		cost_input_batch_mtoken: float = 1.25,                    # The cost per million input tokens with Batch API
		cost_output_direct_mtoken: float = 10.00,                 # The cost per million direct output tokens (1M tokens ~ 750K words)
		cost_output_batch_mtoken: float = 5.00,                   # The cost per million output tokens with Batch API
		assumed_completion_ratio: Optional[float] = 0.5,          # How many output tokens (including both reasoning and visible tokens) to assume will be generated for each request on average (as a ratio of the max_completion_tokens or max_tokens specified for each request, or as a ratio of a default value of 2048 if neither is specified)

		max_task_requests: int = 10000000,                        # Maximum number of requests allowed for an entire task
		max_task_ktokens: int = 5000000,                          # Maximum allowed total number of input tokens (in units of 1000) for an entire task
		max_task_cost: float = 1000.0,                            # Maximum allowed total cost (input + assumed output tokens) of an entire task
		max_session_requests: int = 5000000,                      # Maximum number of requests allowed in a session
		max_session_ktokens: int = 2500000,                       # Maximum allowed total number of input tokens (in units of 1000) in a session
		max_session_cost: float = 500.0,                          # Maximum allowed total cost (input + assumed output tokens) in a session
		max_direct_requests: int = 50,                            # Maximum number of requests allowed in a virtual batch when using the direct API
		max_direct_ktokens: int = 500,                            # Maximum number of input tokens (in units of 1000) to include in a single virtual batch when using the direct API
		max_direct_cost: float = 5.0,                             # Maximum allowed cost (input + assumed output tokens) of a virtual batch when using the direct API
		min_batch_requests: int = 101,                            # Minimum number of requests in a batch in order to use the batch API (otherwise automatically fall back to the direct API)
		max_batch_requests: int = 50000,                          # Maximum number of requests allowed in a batch
		max_batch_mb: int = 100,                                  # Maximum allowed batch size in MB (not MiB)
		max_batch_ktokens: int = 20000,                           # Maximum number of input tokens (in units of 1000) to include in a single batch
		max_batch_cost: float = 50.0,                             # Maximum allowed cost (input + assumed output tokens) of a batch
		max_unpushed_batches: int = 5,                            # Maximum number of unpushed local batches at any one time
		max_remote_batches: int = 15,                             # Maximum number of remote batches at any one time (0 = Only prepare local batches and don't push any yet)
		max_remote_requests: int = 5000000,                       # Maximum number of requests across all uploaded remote batches at any one time
		max_remote_mb: int = 10000,                               # Maximum allowed total size in MB (not MiB) of all uploaded remote batches at any one time
		max_remote_ktokens: int = 60000,                          # Maximum allowed total number of input tokens (in units of 1000) across all uploaded remote batches at any one time
		max_remote_cost: float = 150.0,                           # Maximum allowed cost (input + assumed output tokens) across all uploaded remote batches at any one time
		max_mb_safety: float = 1.01,                              # Safety factor (SF) to use when comparing MB sizes to specified maximum values (can be useful to ensure that server-side MB limits are never used down to the very last byte, as the server could have some fuzzy exact limits, e.g. due to conversions or implicit metadata or overhead, despite giving an exact number for the size limit)
		max_token_safety: float = 1.05,                           # Safety factor (SF) to use when comparing token counts to specified maximum values (token counts are ultimately approximations until the batch is actually executed, so a safety factor can be useful in ensuring that token limits are truly never exceeded in practice)
		force_direct: bool = False,                               # Whether to force all batches to be executed using the direct API instead of the batch API

		warn_predicted_input_factor: float = 1.2,                 # Warn if the predicted number of input tokens deviates from the true number in excess of this multiplicative factor across a batch (>=1.0)
		warn_assumed_completion_factor: float = 1.6,              # Warn if the assumed number of completion tokens deviates from the true number in excess of this multiplicative factor across a batch (>=1.0)
		min_pass_ratio: float = 0.5,                              # If the pass ratio of a batch (number of successful or expired responses as a ratio of the number of requests) is strictly less than this or zero, then the batch is considered as not passed (pass failure)
		max_pass_failures: int = 2,                               # Trigger a processing error if this many consecutive batches do not pass
	):

		self.working_dir = os.path.abspath(working_dir)
		working_dir_parent = os.path.dirname(self.working_dir)
		if not os.path.isdir(working_dir_parent):
			raise ValueError(f"Parent directory of GPT working directory needs to exist: {working_dir_parent}")

		self.name_prefix = name_prefix
		if not self.name_prefix:
			raise ValueError("Name prefix cannot be empty")

		self.autocreate_working_dir = autocreate_working_dir
		created_working_dir = False
		if self.autocreate_working_dir:
			with contextlib.suppress(FileExistsError):
				os.mkdir(self.working_dir)
				created_working_dir = True

		log.info(f"Using GPT requester of prefix '{self.name_prefix}' in dir: {self.working_dir}{' [CREATED]' if created_working_dir else ''}")
		if not os.path.isdir(self.working_dir):
			raise FileNotFoundError(f"GPT working directory does not exist: {self.working_dir}")

		self.dryrun = dryrun
		if self.dryrun:
			log.warning(f"{DRYRUN}GPT requester dry run mode => Not allowing remote batches or writing of state updates and such")

		self.only_process = only_process  # Note: In only-process mode, it is still the responsibility of the task not to add/commit requests (as this functionality is needed for processing request retries it cannot be completely disabled)
		self.wipe_requests = wipe_requests
		self.wipe_task = wipe_task

		self.process_failed_batches = process_failed_batches
		self.retry_fatal_requests = retry_fatal_requests
		if self.process_failed_batches > 0:
			log.info(f"Processing and clearing up to {self.process_failed_batches} failed batches per session")
		elif self.process_failed_batches == 0:
			log.info("An exception will be raised if failed batches occur")
		else:
			log.info("Processing and clearing up any failed batches that occur")

		self.client = client or openai.OpenAI(api_key=openai_api_key, organization=openai_organization, project=openai_project, base_url=client_base_url, **(client_kwargs or {}))
		self.endpoint = endpoint
		if self.endpoint is None:
			self.endpoint = os.environ.get("OPENAI_ENDPOINT")
		if self.endpoint is None:
			self.endpoint = DEFAULT_ENDPOINT
		log.info(f"Using client base URL '{self.client.base_url}' with endpoint '{self.endpoint}'")
		if self.client.base_url.host == 'api.openai.com':
			log.info("View the OpenAI rate/usage limits: https://platform.openai.com/settings/organization/limits")
			log.info("Manage OpenAI file storage: https://platform.openai.com/storage")
			log.info("Manage OpenAI batches: https://platform.openai.com/batches")
			log.info("Monitor the OpenAI usage: https://platform.openai.com/settings/organization/usage")

		self.wandb = wandb
		self.wandb_dir = wandb_dir
		self.wandb_entity = wandb_entity
		self.wandb_project = wandb_project
		self.wandb_run_id = wandb_run_id
		self.wandb_name = wandb_name
		self.wandb_group = wandb_group

		if not isinstance(direct_verbose, DirectVerboseMode):
			self.direct_verbose = DirectVerboseMode.from_str(direct_verbose)
		if self.direct_verbose == DirectVerboseMode.ERROR:
			log.info("Showing verbose requests and responses for all direct requests that have an error")
		elif self.direct_verbose == DirectVerboseMode.WARN:
			log.info("Showing verbose requests and responses for all direct requests that have an error or warning")
		elif self.direct_verbose == DirectVerboseMode.ALWAYS:
			log.info("Showing verbose requests and responses for all direct requests")

		self.show_warnings = show_warnings
		if self.show_warnings < 1:
			raise ValueError(f"Number of warnings to show/log must be at least 1: {self.show_warnings}")
		self.show_errors = show_errors
		if self.show_errors < 1:
			raise ValueError(f"Number of errors to show/log must be at least 1: {self.show_errors}")
		log.info(f"Showing up to {self.show_warnings} warnings and {self.show_errors} errors explicitly per batch per type")

		self.max_retries = max_retries
		if self.max_retries > 0:
			log.info(f"Allowing up to {self.max_retries} retries by default for each request")
		elif self.max_retries == 0:
			log.info("Request retrying is disabled by default")
		else:
			raise ValueError(f"Maximum number of retries cannot be negative: {self.max_retries}")

		self.auto_parse = auto_parse
		log.info(f"Auto-parse is {'enabled' if self.auto_parse else 'disabled'}")

		self.cost_input_direct_mtoken = cost_input_direct_mtoken
		self.cost_input_cached_mtoken = cost_input_cached_mtoken
		self.cost_input_batch_mtoken = cost_input_batch_mtoken
		self.cost_output_direct_mtoken = cost_output_direct_mtoken
		self.cost_output_batch_mtoken = cost_output_batch_mtoken
		if self.cost_input_direct_mtoken < 0 or self.cost_input_cached_mtoken < 0 or self.cost_input_batch_mtoken < 0 or self.cost_output_direct_mtoken < 0 or self.cost_output_batch_mtoken < 0:
			raise ValueError(f"Costs cannot be negative: {self.cost_input_direct_mtoken:.3f}, {self.cost_input_cached_mtoken:.3f}, {self.cost_input_batch_mtoken:.3f}, {self.cost_output_direct_mtoken:.3f}, {self.cost_output_batch_mtoken:.3f}")
		log.info(f"Costs per input mtoken: Direct {self.cost_input_direct_mtoken:.3f}, Cached {self.cost_input_cached_mtoken:.3f}, Batch {self.cost_input_batch_mtoken:.3f}")
		log.info(f"Costs per output mtoken: Direct {self.cost_output_direct_mtoken:.3f}, Batch {self.cost_output_batch_mtoken:.3f}")

		self.W = utils.WandbRun()
		self.token_estimator = tokens.TokenEstimator(assumed_completion_ratio=assumed_completion_ratio, warn=token_estimator_warn)
		self.set_assumed_completion_ratio(assumed_completion_ratio)
		self.token_coster = tokens.TokenCoster(cost_input_direct_mtoken=self.cost_input_direct_mtoken, cost_input_cached_mtoken=self.cost_input_cached_mtoken, cost_input_batch_mtoken=self.cost_input_batch_mtoken, cost_output_direct_mtoken=self.cost_output_direct_mtoken, cost_output_batch_mtoken=self.cost_output_batch_mtoken)
		self.lock = utils.LockFile(path=os.path.join(self.working_dir, f"{self.name_prefix}.lock"), timeout=lock_timeout, poll_interval=lock_poll_interval, status_interval=lock_status_interval)
		self.state = StateFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_state.json"), endpoint=self.endpoint, dryrun=self.dryrun, W=self.W)
		self.queue = QueueFile(path=os.path.join(self.working_dir, f"{self.name_prefix}_queue.jsonl"), endpoint=self.endpoint, token_estimator=self.token_estimator, token_coster=self.token_coster, dryrun=self.dryrun, W=self.W)

		self.remote_update_interval = remote_update_interval
		if self.remote_update_interval < 1.0:
			raise ValueError(f"Remote batch update interval must be at least 1.0s: {self.remote_update_interval:.3g}")
		self.wait_save_interval = wait_save_interval
		if self.wait_save_interval < 10.0:
			raise ValueError(f"State save interval while waiting for batches must be at least 10.0s: {self.wait_save_interval:.3g}")

		self.max_task_requests = max_task_requests
		if self.max_task_requests < 0:
			raise ValueError(f"Maximum number of requests in a task must be at least 0: {self.max_task_requests}")
		self.max_task_ktokens = max_task_ktokens
		if self.max_task_ktokens < 0:
			raise ValueError(f"Maximum task ktokens must be at least 0: {self.max_task_ktokens}")
		self.max_task_cost = max_task_cost
		if self.max_task_cost < 0:
			raise ValueError(f"Maximum task cost must be at least 0: {self.max_task_cost:.3f}")
		self.max_session_requests = max_session_requests
		if self.max_session_requests < 0:
			raise ValueError(f"Maximum number of requests in a session must be at least 0: {self.max_session_requests}")
		self.max_session_ktokens = max_session_ktokens
		if self.max_session_ktokens < 0:
			raise ValueError(f"Maximum session ktokens must be at least 0: {self.max_session_ktokens}")
		self.max_session_cost = max_session_cost
		if self.max_session_cost < 0:
			raise ValueError(f"Maximum session cost must be at least 0: {self.max_session_cost:.3f}")
		self.max_direct_requests = max_direct_requests
		if self.max_direct_requests < 1:
			raise ValueError(f"Maximum number of requests in a direct virtual batch must be at least 1: {self.max_direct_requests}")
		self.max_direct_ktokens = max_direct_ktokens
		if self.max_direct_ktokens < 1:
			raise ValueError(f"Maximum direct virtual batch ktokens must be at least 1: {self.max_direct_ktokens}")
		self.max_direct_cost = max_direct_cost
		if self.max_direct_cost < 0.01:
			raise ValueError(f"Maximum direct virtual batch cost must be at least 0.01: {self.max_direct_cost}")
		self.min_batch_requests = min_batch_requests
		if self.min_batch_requests < 1:
			raise ValueError(f"Minimum number of requests in a remote batch must be at least 1: {self.min_batch_requests}")
		self.max_batch_requests = max_batch_requests
		if self.max_batch_requests < 1:
			raise ValueError(f"Maximum number of requests in a batch must be at least 1: {self.max_batch_requests}")
		self.max_batch_mb = max_batch_mb
		if self.max_batch_mb < 1:
			raise ValueError(f"Maximum batch size in MB must be at least 1: {self.max_batch_mb}")
		self.max_batch_ktokens = max_batch_ktokens
		if self.max_batch_ktokens < 1:
			raise ValueError(f"Maximum batch ktokens must be at least 1: {self.max_batch_ktokens}")
		self.max_batch_cost = max_batch_cost
		if self.max_batch_cost < 0.01:
			raise ValueError(f"Maximum batch cost must be at least 0.01: {self.max_batch_cost:.3f}")
		self.max_unpushed_batches = max_unpushed_batches
		if self.max_unpushed_batches < 1:
			raise ValueError(f"Maximum number of unpushed batches must be at least 1: {self.max_unpushed_batches}")
		self.max_remote_batches = max_remote_batches
		if self.max_remote_batches < 0:
			raise ValueError(f"Maximum number of remote batches must be at least 0: {self.max_remote_batches}")
		self.max_remote_requests = max_remote_requests
		if self.max_remote_requests < self.max_batch_requests:
			raise ValueError(f"Maximum number of requests across all remote batches must be at least as great as the maximum number of requests in a batch: {self.max_remote_requests} vs {self.max_batch_requests}")
		self.max_remote_mb = max_remote_mb
		if self.max_remote_mb < self.max_batch_mb:
			raise ValueError(f"Maximum total uploaded remote batch size in MB must be at least as great as the maximum batch size in MB: {self.max_remote_mb} vs {self.max_batch_mb}")
		self.max_remote_ktokens = max_remote_ktokens
		if self.max_remote_ktokens < self.max_batch_ktokens:
			raise ValueError(f"Maximum total uploaded remote batch ktokens must be at least as great as the maximum batch ktokens: {self.max_remote_ktokens} vs {self.max_batch_ktokens}")
		self.max_remote_cost = max_remote_cost
		if self.max_remote_cost < self.max_batch_cost:
			raise ValueError(f"Maximum total uploaded remote batch cost must be at least as great as the maximum batch cost: {self.max_remote_cost:.3f} vs {self.max_batch_cost:.3f}")
		self.max_mb_safety = max_mb_safety
		if self.max_mb_safety < 1.0:
			raise ValueError(f"The number of MB safety factor must be at least 1.0: {self.max_mb_safety:.3g}")
		self.max_token_safety = max_token_safety
		if self.max_token_safety < 1.0:
			raise ValueError(f"The number of tokens safety factor must be at least 1.0: {self.max_token_safety:.3g}")
		self.force_direct = force_direct

		log.info(f"Using safety factors (SF) of {self.max_mb_safety:.3g} for MB size, {self.max_token_safety:.3g} for tokens")
		log.info(f"Using direct batch limits of {self.max_direct_requests} requests, {self.max_direct_tokens} tokens, {self.max_direct_cost:.3f} assumed cost")
		log.info(f"Using batch size limits of {self.max_batch_requests} requests, {utils.format_size_si(self.max_batch_size)}, {self.max_batch_tokens} tokens, {self.max_batch_cost:.3f} assumed cost")
		log.info(f"Using remote size limits of {self.max_remote_requests} requests, {utils.format_size_si(self.max_remote_size)}, {self.max_remote_tokens} tokens, {self.max_remote_cost:.3f} assumed cost")
		if self.force_direct:
			log.info("Using direct API for all batches (FORCED)")
		else:
			log.info(f"Using direct API for all batches with at most {self.min_batch_requests - 1} requests")
		log.info(f"Allowing at most {self.max_unpushed_batches} unpushed and {self.max_remote_batches} remote batches at once")
		log.info(f"Allowing at most {self.max_session_requests} requests, {self.max_session_tokens} tokens, {self.max_session_cost:.3f} assumed cost per session")
		log.info(f"Allowing at most {self.max_task_requests} requests, {self.max_task_tokens} tokens, {self.max_task_cost:.3f} assumed cost per task")

		self.warn_predicted_input_factor = warn_predicted_input_factor
		if self.warn_predicted_input_factor < 1.0:
			raise ValueError(f"Predicted input tokens factor to warn at must be at least 1.0: {self.warn_predicted_input_factor:.3g}")
		self.warn_assumed_completion_factor = warn_assumed_completion_factor
		if self.warn_assumed_completion_factor < 1.0:
			raise ValueError(f"Assumed completion tokens factor to warn at must be at least 1.0: {self.warn_assumed_completion_factor:.3g}")
		self.min_pass_ratio = min_pass_ratio
		if not 0.0 <= self.min_pass_ratio <= 1.0:
			raise ValueError(f"Minimum pass ratio must be in the interval [0,1]: {self.min_pass_ratio:.3g}")
		self.max_pass_failures = max_pass_failures
		if self.max_pass_failures < 1:
			raise ValueError(f"Maximum number of pass failures must be at least 1: {self.max_pass_failures}")
		log.info(f"Warning on batches that exceed a predicted input tokens factor of {self.warn_predicted_input_factor:.3g} or an assumed completion tokens factor of {self.warn_assumed_completion_factor:.3g}")
		log.info(f"Triggering a processing error if {self.max_pass_failures} consecutive batches have a pass ratio less than {self.min_pass_ratio:.3g}")

		self._enter_stack = contextlib.ExitStack()
		self.type_cache: Optional[utils.SerialTypeCache] = None
		self.S: Optional[State] = None
		self.PQ: Optional[PoolQueue] = None
		self.P: Optional[list[CachedGPTRequest]] = None
		self.Q: Optional[list[Optional[CachedGPTRequest]]] = None
		self.max_request_id: Optional[int] = None
		self.max_direct_batch_id: Optional[int] = None
		self.session_push_stats: Optional[PushStats] = None
		self.session_processed_failed_batches: Optional[int] = None

	def set_assumed_completion_ratio(self, assumed_completion_ratio: Optional[float]):
		self.assumed_completion_ratio = assumed_completion_ratio
		self.token_estimator.set_assumed_completion_ratio(assumed_completion_ratio)
		if self.assumed_completion_ratio is not None:
			if not 0.0 <= self.assumed_completion_ratio <= 1.0:
				raise ValueError(f"Assumed output completion ratio must be in the interval [0,1]: {self.assumed_completion_ratio:.3g}")
			log.info(f"Assuming an output token completion ratio of {self.assumed_completion_ratio:.3g}")

	@property
	def max_task_tokens(self) -> int:
		# Returns the maximum number of tokens allowed in a task (including possible reduction due to the token safety factor)
		return max(round(self.max_task_ktokens * 1000 / self.max_token_safety), 0)

	@property
	def max_session_tokens(self) -> int:
		# Returns the maximum number of tokens allowed in a session (including possible reduction due to the token safety factor)
		return max(round(self.max_session_ktokens * 1000 / self.max_token_safety), 0)

	@property
	def max_direct_tokens(self) -> int:
		# Returns the maximum number of tokens allowed in a direct virtual batch (including possible reduction due to the token safety factor)
		return max(round(self.max_direct_ktokens * 1000 / self.max_token_safety), 1000)

	@property
	def max_batch_size(self) -> int:
		# Returns the maximum number of bytes allowed in a batch (including possible reduction due to the MB safety factor)
		return max(round(self.max_batch_mb * 1000000 / self.max_mb_safety), 1000000)

	@property
	def max_batch_tokens(self) -> int:
		# Returns the maximum number of tokens allowed in a batch (including possible reduction due to the token safety factor)
		return max(round(self.max_batch_ktokens * 1000 / self.max_token_safety), 1000)

	@property
	def max_remote_size(self) -> int:
		# Returns the maximum number of bytes allowed across all remote batches (including possible reduction due to the MB safety factor)
		return max(round(self.max_remote_mb * 1000000 / self.max_mb_safety), 1000000)

	@property
	def max_remote_tokens(self) -> int:
		# Returns the maximum number of tokens allowed across all remote batches (including possible reduction due to the token safety factor)
		return max(round(self.max_remote_ktokens * 1000 / self.max_token_safety), 1000)

	# Configure an argparse parser to incorporate an argument group for the keyword arguments that can be passed to the init of this class
	@staticmethod
	def configure_argparse(
		parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup],  # noqa / Argument parser or group
		*,                                                                # Keyword arguments only beyond here
		title: Optional[str] = 'GPT requester',                           # If parser is not already an argument group, the title to use for the created argument group
		description: Optional[str] = None,                                # If parser is not already an argument group, the description to use for the created argument group
		group_kwargs: Optional[dict[str, Any]] = None,                    # If parser is not already an argument group, the extra keyword arguments to use for the created argument group
		include_endpoint: bool = False,                                   # Whether to include an argument for the API endpoint to use (often this should be task-specific and more specific arguments like chat_endpoint should be defined and used by the task itself)
		include_auto_parse: bool = False,                                 # Whether to include an argument for auto-parsing (often this is task-specific)
		include_completion_ratio: bool = False,                           # Whether to include an argument for the assumed completion ratio (often this is task-specific)
		**defaults,                                                       # noqa / Keyword arguments that can be used to override individual default argument values
	) -> argparse._ArgumentGroup:                                         # noqa / Returns the passed or newly created argument group

		group = parser.add_argument_group(title=title, description=description, **(group_kwargs if group_kwargs is not None else {})) if isinstance(parser, argparse.ArgumentParser) else parser
		add_argument = functools.partial(utils.add_init_argparse, cls=GPTRequester, parser=group, defaults=defaults)

		add_argument(name='dryrun', help="Whether to perform a dry run (e.g. no OpenAI API calls, no pushing batches to remote, no processing finished batches, no writing to disk, ...)")
		add_argument(name='only_process', help="Whether to solely process existing unfinished remote batches and not generate/commit/push anything new")
		add_argument(name='process_failed_batches', metavar='MAXNUM', help="Whether to force the processing of any failed batches, thereby finalizing them and clearing them from the remote and pipeline (<0 = Process all failed batches, 0 = Do not process any failed batches, >0 = Process at most N failed batches in this session)")
		add_argument(name='retry_fatal_requests', help="Whether to retry fatal requests from failed batches that are processed, or otherwise skip and fail them")
		add_argument(name='wipe_requests', help="CAUTION: Wipe and forget all ongoing requests and request batches without processing them (cancels/deletes batches from remote, does not affect any already finished/processed requests, consider running the requester with --only_process prior to wiping)")
		add_argument(name='wipe_task', help="CAUTION: Wipe and forget all progress and existing results on the task, and start completely afresh from scratch (implies wipe_requests first, also triggers task manager actions if requester is being managed by one)")

		add_argument(name='openai_api_key', metavar='KEY', help="OpenAI API key (see openai.OpenAI, ends up in request headers)")
		add_argument(name='openai_organization', metavar='ID', help="OpenAI organization (see openai.OpenAI, ends up in request headers)")
		add_argument(name='openai_project', metavar='ID', help="OpenAI project (see openai.OpenAI, ends up in request headers)")
		add_argument(name='client_base_url', metavar='URL', help="Base URL to use for the OpenAI API client (see openai.OpenAI, servers other than OpenAI's servers can be configured to expose an OpenAI API with suitable endpoints)")
		if include_endpoint:
			add_argument(name='endpoint', help="API endpoint to use for all requests (Careful: May be ignored by specific tasks that require specific endpoints)")

		add_argument(name='wandb', default=None, help="Enable wandb logging")
		add_argument(name='wandb', default=True, help="Disable wandb logging")
		add_argument(name='wandb_dir', metavar='DIR', help="Wandb directory (must already exist, see also the WANDB_DIR environment variable)")
		add_argument(name='wandb_entity', metavar='ENTITY', help="Wandb entity")
		add_argument(name='wandb_project', metavar='PROJ', help="Wandb project name")
		add_argument(name='wandb_run_id', metavar='ID', help="Wandb run ID to create or resume (auto-generated run IDs are typically lowercase alphanumeric strings of length 8)")
		add_argument(name='wandb_name', metavar='NAME', help="Wandb run name")
		add_argument(name='wandb_group', metavar='GROUP', help="Wandb group (defaults to the class name of the task manager class)")

		add_argument(name='autocreate_working_dir', help="Do not automatically create the GPT working directory if it does not exist (parent directory must already exist)")
		add_argument(name='lock_timeout', metavar='SEC', unit='s', help="Timeout (if any) to use when attempting to lock exclusive access to the files in the GPT working directory corresponding to the given name prefix (see utils.LockFile)")
		add_argument(name='lock_poll_interval', metavar='SEC', unit='s', help="Lock file polling interval (see utils.LockFile)")
		add_argument(name='lock_status_interval', metavar='SEC', unit='s', help="Lock file status update interval (see utils.LockFile)")
		add_argument(name='token_estimator_warn', metavar='MODE', help="Warning mode to use for internal token estimator (see tokens.TokenEstimator)")
		add_argument(name='remote_update_interval', metavar='SEC', unit='s', help="Interval in multiples of which to update remote batch states when waiting for remote batches to finish")
		add_argument(name='wait_save_interval', metavar='SEC', unit='s', help="Minimum time interval in seconds between successive state saves to disk while waiting for batches to complete")

		add_argument(name='direct_verbose', metavar='MODE', help="Whether to show verbose requests/responses when using the direct API (options: never, error, warn (i.e. if a warning or error occurs), always)")
		add_argument(name='show_warnings', metavar='NUM', help="How many warnings to show/log per batch per warning type when processing responses")
		add_argument(name='show_errors', metavar='NUM', help="How many errors to show/log per batch per error type when processing responses")
		add_argument(name='max_retries', metavar='NUM', help="Maximum number of retries to allow by default for a request (e.g. 2 retries => 3 attempts allowed, retries due to batch cancellation or expiration do not count towards the retry count)")
		if include_auto_parse:
			add_argument(name='auto_parse', help="Whether to perform auto-parsing to help validate and Python-ify API requests and responses")

		add_argument(name='cost_input_direct_mtoken', metavar='COST', help="The cost per million direct input tokens (1M tokens ~ 750K words)")
		add_argument(name='cost_input_cached_mtoken', metavar='COST', help="The cost per million cached input tokens")
		add_argument(name='cost_input_batch_mtoken', metavar='COST', help="The cost per million input tokens with Batch API")
		add_argument(name='cost_output_direct_mtoken', metavar='COST', help="The cost per million direct output tokens (1M tokens ~ 750K words)")
		add_argument(name='cost_output_batch_mtoken', metavar='COST', help="The cost per million output tokens with Batch API")
		if include_completion_ratio:
			add_argument(name='assumed_completion_ratio', metavar='RATIO', help="How many output tokens (including both reasoning and visible tokens) to assume will be generated for each request on average (as a ratio of the max_completion_tokens or max_tokens specified for each request, or as a ratio of a default value of 2048 if neither is specified)")

		add_argument(name='max_task_requests', metavar='NUM', help="Maximum number of requests allowed for an entire task")
		add_argument(name='max_task_ktokens', metavar='KTOK', unit=' ktokens', help="Maximum allowed total number of input tokens (in units of 1000) for an entire task")
		add_argument(name='max_task_cost', metavar='COST', help="Maximum allowed total cost (input + assumed output tokens) of an entire task")
		add_argument(name='max_session_requests', metavar='NUM', help="Maximum number of requests allowed in a session")
		add_argument(name='max_session_ktokens', metavar='KTOK', unit=' ktokens', help="Maximum allowed total number of input tokens (in units of 1000) in a session")
		add_argument(name='max_session_cost', metavar='COST', help="Maximum allowed total cost (input + assumed output tokens) in a session")
		add_argument(name='max_direct_requests', metavar='NUM', help="Maximum number of requests allowed in a virtual batch when using the direct API")
		add_argument(name='max_direct_ktokens', metavar='KTOK', unit=' ktokens', help="Maximum number of input tokens (in units of 1000) to include in a single virtual batch when using the direct API")
		add_argument(name='max_direct_cost', metavar='COST', help="Maximum allowed cost (input + assumed output tokens) of a virtual batch when using the direct API")
		add_argument(name='min_batch_requests', metavar='NUM', help="Minimum number of requests in a batch in order to use the batch API (otherwise fall back to the direct API)")
		add_argument(name='max_batch_requests', metavar='NUM', help="Maximum number of requests allowed in a batch")
		add_argument(name='max_batch_mb', metavar='MB', unit='MB', help="Maximum allowed batch size in MB (not MiB)")
		add_argument(name='max_batch_ktokens', metavar='KTOK', unit=' ktokens', help="Maximum number of input tokens (in units of 1000) to include in a single batch")
		add_argument(name='max_batch_cost', metavar='COST', help="Maximum allowed cost (input + assumed output tokens) of a batch")
		add_argument(name='max_unpushed_batches', metavar='NUM', help="Maximum number of unpushed local batches at any one time")
		add_argument(name='max_remote_batches', metavar='NUM', help="Maximum number of remote batches at any one time (0 = Only prepare local batches and don't push any yet)")
		add_argument(name='max_remote_requests', metavar='NUM', help="Maximum number of requests across all uploaded remote batches at any one time")
		add_argument(name='max_remote_mb', metavar='MB', unit='MB', help="Maximum allowed total size in MB (not MiB) of all uploaded remote batches at any one time")
		add_argument(name='max_remote_ktokens', metavar='KTOK', unit=' ktokens', help="Maximum allowed total number of input tokens (in units of 1000) across all uploaded remote batches at any one time")
		add_argument(name='max_remote_cost', metavar='COST', help="Maximum allowed cost (input + assumed output tokens) across all uploaded remote batches at any one time")
		add_argument(name='max_mb_safety', metavar='FACTOR', help="Safety factor (SF) to use when comparing MB sizes to specified maximum values (can be useful to ensure that server-side MB limits are never used down to the very last byte, as the server could have some fuzzy exact limits, e.g. due to conversions or implicit metadata or overhead, despite giving an exact number for the size limit)")
		add_argument(name='max_token_safety', metavar='FACTOR', help="Safety factor (SF) to use when comparing token counts to specified maximum values (token counts are ultimately approximations until the batch is actually executed, so a safety factor can be useful in ensuring that token limits are truly never exceeded in practice)")
		add_argument(name='force_direct', help="Whether to force all batches to be executed using the direct API instead of the batch API")

		add_argument(name='warn_predicted_input_factor', metavar='FACTOR', help="Warn if the predicted number of input tokens deviates from the true number in excess of this multiplicative factor across a batch (>=1.0)")
		add_argument(name='warn_assumed_completion_factor', metavar='FACTOR', help="Warn if the assumed number of completion tokens deviates from the true number in excess of this multiplicative factor across a batch (>=1.0)")
		add_argument(name='min_pass_ratio', metavar='RATIO', help="If the pass ratio of a batch (number of successful or expired responses as a ratio of the number of requests) is strictly less than this or zero, then the batch is considered as not passed (pass failure)")
		add_argument(name='max_pass_failures', metavar='NUM', help="Trigger a processing error if this many consecutive batches do not pass")

		return group

	# Save the GPT requester state to file
	def save_state(self, rstack: utils.RevertStack, show_log: bool = True, save: bool = True):
		# rstack = RevertStack to use for the safe reversible saving of the state file
		# show_log = Whether to log that the state file was saved
		# save = Whether to actually save the state to file (or just update the log)
		if save:
			self.state.save(rstack=rstack, show_log=show_log)
		else:
			self.state.log_summary(rstack=rstack)
		if isinstance(rstack, utils.LogRevertStack):
			rstack.log({
				'Pushed/session_requests': self.session_push_stats.total_requests,
				'Pushed/session_input_ktok': self.session_push_stats.total_tokens_cost.input_tokens / 1000,
				'Pushed/session_pred_cost': self.session_push_stats.total_tokens_cost.cost,
				'Metrics/processed_failed_batches': self.session_processed_failed_batches,
			})

	# Initialize wandb with a particular configuration
	@classmethod
	def wandb_init(cls, config: Union[dict[str, Any], utils.SupportsVars], **wandb_kwargs) -> ContextManager[Optional[wandb.sdk.wandb_run.Run]]:
		# config = Configuration parameters to use for wandb (_* parameters are filtered out, wandb* parameters are filtered out and used for the call to wandb.init(), config can be a dict, argparse.Namespace, flat omegaconf.DictConfig, etc..)
		# wandb_kwargs = Miscellaneous arguments or no-questions-asked argument overrides for the call to wandb.init()
		# Returns a context manager that can be used to ensure that any newly initialized wandb run eventually receives a call to finish()
		# If the config contains a parameter called 'wandb' that is False or None, then no wandb.init() call is made and a null context manager is returned that just returns None
		# Otherwise if a wandb run has previously been initialized in this process, then no wandb.init() call is made and a null context manager is returned that just returns that existing wandb run (i.e. doesn't trigger a call to finish() on exiting, as that's some other code's responsibility)
		# Otherwise, a wandb.init() call is made based on the passed config and wandb_kwargs, and the newly initialized run is returned (Run's are inherently context managers that return themselves on entering, and ensure a call to finish() on exiting)

		config_type = type(config)
		if isinstance(config, dict):
			cfg = config.copy()
		elif config_type.__name__ == 'DictConfig' and f"{config_type.__module__}.{config_type.__qualname__}".startswith('omegaconf.'):
			cfg = {key: getattr(config, key) for key in dir(config)}
		else:
			cfg = dict(vars(config))

		wandb_cfg = {}
		for key in tuple(cfg):
			if key and key[0] == '_':
				del cfg[key]
			elif key.startswith('wandb'):
				wandb_cfg[key] = cfg.pop(key)

		if not wandb_cfg.get('wandb', True):
			return contextlib.nullcontext(None)
		elif wandb.run is not None:
			return contextlib.nullcontext(wandb.run)
		else:
			wandb_init_kwargs = dict(
				entity=wandb_cfg.get('wandb_entity'),
				project=wandb_cfg.get('wandb_project'),
				dir=wandb_cfg.get('wandb_dir'),
				id=wandb_cfg.get('wandb_run_id'),
				name=wandb_cfg.get('wandb_name'),
				config=cfg,
				allow_val_change=False,
				group=wandb_cfg.get('wandb_group'),
				job_type='gpt_batch_api',
				mode='online',
				force=True,
				reinit=False,
				resume='allow',
			)
			wandb_init_kwargs.update(wandb_kwargs)
			return wandb.init(**wandb_init_kwargs)

	# Configure wandb as appropriate based on class instance(s) that use @utils.init_kwargs
	@contextlib.contextmanager
	def configure_wandb(self, *init_kwargs_objs: Union[Any, tuple[Any, dict[str, Any]]], **wandb_kwargs) -> ContextManager[None]:
		# init_kwargs_objs = Objects to extract configuration parameters from (must be instances of classes that use @utils.init_kwargs), possibly packed into a tuple along with a dict of custom extra wandb configuration parameters
		# wandb_kwargs = Extra wandb keyword arguments or argument overrides to pass to self.wandb_init()
		# Returns a context manager that ensures wandb is initialized if required, and that updates self.W with any newly initialized wandb run
		# Configuration parameters are extracted from each object obj based on obj.__kwargs__ for the keys, and getattr(obj, key) for the values (extra parameters or parameter value overrides can be specified using a dict of custom extra wandb configuration parameters)
		# The call to this method does not do anything if a previous call to this method was already made and successfully initialized wandb, or if dry run is active

		if self.W or self.dryrun:
			yield
			return

		config = {}
		for item in init_kwargs_objs:

			if isinstance(item, tuple):
				obj, extra_configs = item
			else:
				obj = item
				extra_configs = {}

			for name, init_kwarg in obj.__kwargs__.items():
				if init_kwarg.base_type in (str, int, float, bool) and name not in extra_configs:
					try:
						value = getattr(obj, name)
					except AttributeError:
						pass
					else:
						if value is not None and not isinstance(value, init_kwarg.base_type):
							if init_kwarg.base_type is float and isinstance(value, int):
								value = init_kwarg.base_type(value)
							elif isinstance(value, enum.Enum):
								value = value.name
							else:
								value = format(value)  # If we don't know what's going on then formatting to a string seems safest (rather than for instance arbitrarily applying bool() or int() on a miscellaneous type)
						if name in config:
							raise ValueError(f"Encountered multiple configuration parameters of the same name: {name}")
						config[name] = value

			config.update(extra_configs)

		try:
			with self.wandb_init(config=config, **wandb_kwargs) as run:
				self.W.set(run=run)
				yield
		finally:
			self.W.reset()

	# Custom extra wandb configuration parameters
	@property
	def extra_wandb_configs(self) -> dict[str, Any]:
		return dict(
			client_base_url=self.client.base_url,
			lock_timeout=self.lock.timeout,
			lock_poll_interval=self.lock.poll_interval,
			lock_status_interval=self.lock.status_interval,
			token_estimator_warn=self.token_estimator.warn,
		)

	# Enter method for the required use of GPTRequester as a context manager
	def __enter__(self) -> Self:
		with self._enter_stack as enter_stack:
			enter_stack.enter_context(self.lock)
			enter_stack.enter_context(self.configure_wandb((self, self.extra_wandb_configs)))
			enter_stack.callback(self.on_exit)
			self.type_cache = utils.SerialTypeCache()
			enter_stack.enter_context(self.state)
			self.S = self.state.state
			self.max_request_id = self.S.max_request_id
			self.max_direct_batch_id = self.S.max_direct_batch_id
			enter_stack.enter_context(self.queue)
			self.PQ = self.queue.pool_queue
			self.P = self.PQ.pool
			self.Q = self.PQ.queue
			self.session_push_stats = PushStats()
			self.session_processed_failed_batches = 0
			self.validate_state_queue(clean=True)
			self.wipe(wipe_requests=self.wipe_requests, wipe_task=self.wipe_task)  # Only does something if one of the arguments is True
			self._enter_stack = enter_stack.pop_all()
		assert self._enter_stack is not enter_stack
		return self

	# Exit method for the required use of GPTRequester as a context manager
	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		if self.P:
			log.warning(f"Exiting GPT requester with {len(self.P)} uncommitted requests in the request pool")
		return self._enter_stack.__exit__(exc_type, exc_val, exc_tb)

	# Local actions to perform on exit
	def on_exit(self):
		self.type_cache = None
		self.S = self.PQ = self.P = self.Q = None
		self.max_request_id = None
		self.max_direct_batch_id = None
		self.session_push_stats = None
		self.session_processed_failed_batches = None

	# Perform a wipe of requests and/or the entire task
	def wipe(self, wipe_requests: bool, wipe_task: bool):
		# wipe_requests = Whether to wipe all ongoing requests and batches
		# wipe_task = Whether to wipe the complete requester state (as pertaining to the task)
		# Wiping is NOT a revertible operation, and an indeterminate/inconsistent state in memory/on disk/on remote may result if an exception occurs during wiping

		if wipe_requests or wipe_task:
			log.info('\xB7' * 60)
			self.log_status()
			with utils.DelayKeyboardInterrupt():

				log.info('\xB7' * 60)
				log.warning("Wiping all ongoing requests and batches...")

				self.prepare_process_batches()

				unfinished_batches = {batch.remote.batch.id: None for batch in self.S.batches if batch.remote is not None and not batch.remote.finished}
				remote_files = {remote_file_id: None for batch in self.S.batches if batch.remote is not None for remote_file_id in (batch.remote.batch.error_file_id, batch.remote.batch.output_file_id, batch.remote.file.id) if remote_file_id is not None}
				local_files = {batch.local_jsonl: None for batch in self.S.batches}

				self.P.clear()
				self.Q.clear()
				self.S.max_request_id = self.max_request_id
				self.S.queue.request_id_meta = self.PQ.queue_request_id_meta()
				self.S.batches.clear()
				self.S.num_pass_failures = 0
				self.session_processed_failed_batches = 0

				if not wipe_task:
					with utils.LogRevertStack(W=self.W) as rstack:
						self.validate_state_queue(clean=True)
						self.queue.save(rstack=rstack)
						self.save_state(rstack=rstack)

				for batch_id in unfinished_batches:
					self.cancel_remote_batch(batch_id=batch_id)
				for remote_file_id in remote_files:
					self.delete_remote_file(file_id=remote_file_id)
				for local_file in local_files:
					self.delete_local_file(path=local_file)

				self.wipe_requests = False
				log.warning("Wiped all ongoing requests and batches")

				if wipe_task:

					log.warning("Wiping complete GPT requester state...")

					with utils.LogRevertStack(W=self.W) as rstack:
						self.state.create(rstack=rstack)
						self.S = self.state.state
						self.max_request_id = self.S.max_request_id
						self.max_direct_batch_id = self.S.max_direct_batch_id
						self.queue.create(rstack=rstack)
						self.PQ = self.queue.pool_queue
						self.P = self.PQ.pool
						self.Q = self.PQ.queue
						self.session_push_stats = PushStats()
						self.validate_state_queue(clean=True)

					self.wipe_task = False
					log.warning("Wiped complete GPT requester state")

	# Validate that there are no obvious issues with the current state and queue (clean refers to the expected status right after a commit)
	def validate_state_queue(self, *, clean: bool):
		# clean = Whether the pool-queue status is expected to be clean, i.e. like fresh after a commit where the pool is empty and there are no None entries in the queue

		if clean:
			if self.P:
				raise ValueError("Request pool state is unexpectedly not clean")
			if any(cached_req is None for cached_req in self.Q):
				raise ValueError("Request queue state is unexpectedly not clean")
			if self.S.max_request_id != self.max_request_id:
				raise ValueError(f"Maximum request ID is inconsistent between GPT requester class and state: {self.max_request_id} vs {self.S.max_request_id}")
		if self.max_request_id < self.S.max_request_id:
			raise ValueError(f"Maximum request ID of GPT requester class is less than its state: {self.max_request_id} < {self.S.max_request_id}")

		if self.S.max_direct_batch_id > self.max_direct_batch_id:
			raise ValueError(f"Maximum direct batch ID state is inconsistent: {self.S.max_direct_batch_id} > {self.max_direct_batch_id}")
		if self.S.queue.request_id_meta != self.PQ.queue_request_id_meta():
			raise ValueError("ID-meta map is inconsistent between the state and queue files for the request queue")

		for batch in self.S.batches:
			if batch.id <= 0:
				raise ValueError(f"Invalid batch ID: {batch.id}")
			if batch.id > self.S.max_batch_id:
				raise ValueError(f"Batch ID is greater than the supposed maximum assigned batch ID: {batch.id} > {self.S.max_batch_id}")
			if len(batch.request_info) != batch.num_requests:
				raise ValueError(f"Mismatch in number of requests in batch {batch.id}: {len(batch.request_info)} vs {batch.num_requests}")
			if batch.remote is not None:
				if batch.remote.batch.input_file_id != batch.remote.file.id:
					raise ValueError(f"Batch file ID is not consistent: {batch.remote.batch.input_file_id} vs {batch.remote.file.id}")
			if batch.true_tokens_cost is not None or batch.metrics is not None:
				raise ValueError(f"Batch {batch.id} unexpectedly has non-None true cost and/or metrics")
		if not utils.is_ascending((batch.id for batch in self.S.batches), strict=True):
			raise ValueError(f"The batch IDs are not strictly ascending: {list(batch.id for batch in self.S.batches)}")

		for batch in self.S.batches:
			if not utils.is_ascending(batch.request_info.keys(), strict=True):
				raise ValueError(f"The request IDs in a batch are not strictly ascending: {list(batch.request_info.keys())}")
		if not utils.is_ascending(self.S.queue.request_id_meta.keys(), strict=True):
			raise ValueError(f"The request IDs in the request queue are not strictly ascending: {list(self.S.queue.request_id_meta.keys())}")
		if not utils.is_ascending((pool_request_ids := [cached_req.item.id for cached_req in self.P]), strict=True):
			raise ValueError(f"The request IDs in the request pool are not strictly ascending: {pool_request_ids}")

		for batch in self.S.batch_history:
			if batch.request_info:
				raise ValueError(f"The request IDs in historical batch {batch.id} were not cleared")
			if batch.true_tokens_cost is None or batch.metrics is None:
				raise ValueError(f"Historical batch {batch.id} does not have true cost and metrics")
		batch_id_history = [batch.id for batch in self.S.batch_history]
		if not utils.is_ascending(batch_id_history, strict=True):
			raise ValueError(f"The batch IDs in the batch history are not strictly ascending: {batch_id_history}")

		for batch in self.S.direct_history:
			if batch.request_info:
				raise ValueError(f"The request IDs in historical direct batch {batch.id} were not cleared")
			if batch.true_tokens_cost is None or batch.metrics is None:
				raise ValueError(f"Historical direct batch {batch.id} does not have true cost and metrics")
		direct_id_history = [batch.id for batch in self.S.direct_history]
		if not utils.is_ascending(direct_id_history, strict=True):
			raise ValueError(f"The batch IDs in the direct history are not strictly ascending: {direct_id_history}")

		req_id_blocks = (*(batch.request_info.keys() for batch in self.S.batches), self.S.queue.request_id_meta.keys(), pool_request_ids)
		all_req_ids = collections.Counter(itertools.chain.from_iterable(req_id_blocks))
		if any(count > 1 for count in all_req_ids.values()):
			raise ValueError(f"There are reused request ID(s): {', '.join(f'{count} \xD7 {req_id}' for req_id, count in sorted(all_req_ids.items()) if count > 1)}")
		req_id_intervals = [(min(req_ids, default=None), max(req_ids, default=None)) for req_ids in req_id_blocks]
		flat_req_id_bounds = [req_id for req_id_interval in req_id_intervals for req_id in req_id_interval if req_id is not None]
		if any(req_id > self.max_request_id for req_id in flat_req_id_bounds):
			raise ValueError(f"There are request ID(s) greater than the supposed maximum assigned request ID {self.max_request_id}: {req_id_intervals}")

		if self.S.task_push_stats.total_requests < self.session_push_stats.total_requests:
			raise ValueError(f"Unexpected push stats of task vs session in terms of requests: {self.S.task_push_stats.total_requests} < {self.session_push_stats.total_requests}")
		for field in dataclasses.fields(TokensCost):  # noqa
			if getattr(self.S.task_push_stats.total_tokens_cost, field.name) < getattr(self.session_push_stats.total_tokens_cost, field.name):
				raise ValueError(f"Unexpected push stats of task vs session vs batch state in terms of {field.name}: {getattr(self.S.task_push_stats.total_tokens_cost, field.name)} < {getattr(self.session_push_stats.total_tokens_cost, field.name)}")

		for batch in self.S.batches:
			batch_tokens_cost = TokensCost().add(req_info.tokens_cost for req_info in batch.request_info.values())
			if not batch.tokens_cost.equals(batch_tokens_cost):
				raise ValueError(f"Total tokens cost mismatch for batch: {batch.tokens_cost} vs {batch_tokens_cost}")

	# Log the current GPT requester status
	def log_status(self):
		pool_queue_tokens_cost = TokensCost().add((cached_req.info.tokens_cost for cached_req in self.P), (cached_req.info.tokens_cost for cached_req in self.Q if cached_req is not None))
		local_tokens_cost = TokensCost().add(batch.tokens_cost for batch in self.S.batches if batch.remote is None)
		remote_tokens_cost = TokensCost().add(batch.tokens_cost for batch in self.S.batches if batch.remote is not None)
		log.info(f"There are {self.PQ.pool_len} POOLED requests and {self.PQ.queue_len} QUEUED requests with a combined total of {self.PQ.pool_len + self.PQ.queue_len} requests, {utils.format_size_si(sum(cached_req.info.json_size for cached_req in self.P) + sum(cached_req.info.json_size for cached_req in self.Q if cached_req is not None))}, {pool_queue_tokens_cost.input_tokens} tokens, {pool_queue_tokens_cost.cost_batch:.3f} assumed cost")
		log.info(f"There are {self.num_unpushed_batches()} unpushed LOCAL batches ({self.num_pushable_batches()} pushable and {self.num_unpushable_batches()} unpushable) with a total of {sum(batch.num_requests for batch in self.S.batches if batch.remote is None)} requests, {utils.format_size_si(sum(batch.local_jsonl_size for batch in self.S.batches if batch.remote is None))}, {local_tokens_cost.input_tokens} tokens, {local_tokens_cost.cost_batch:.3f} assumed cost")
		log.info(f"There are {self.num_unfinished_batches()} unfinished and {self.num_finished_batches()} finished REMOTE batches with a combined total of {sum(batch.num_requests for batch in self.S.batches if batch.remote is not None)} requests, {utils.format_size_si(sum(batch.local_jsonl_size for batch in self.S.batches if batch.remote is not None))}, {remote_tokens_cost.input_tokens} tokens, {remote_tokens_cost.cost_batch:.3f} assumed cost")
		log.info(f"SESSION push statistics are {self.session_push_stats.total_requests} requests, {self.session_push_stats.total_tokens_cost.input_tokens} input tokens, {self.session_push_stats.total_tokens_cost.cost:.3f} assumed cost")
		log.info(f"TASK push statistics are {self.S.task_push_stats.total_requests} requests, {self.S.task_push_stats.total_tokens_cost.input_tokens} input tokens, {self.S.task_push_stats.total_tokens_cost.cost:.3f} assumed cost")
		batch_metrics, direct_metrics, all_metrics = self.S.metrics.batch.total, self.S.metrics.direct.total, self.S.metrics.all.total
		log.info(f"{batch_metrics.num_requests} requests have been completed in BATCH mode, entailing {len(batch_metrics.models)} models, {len(batch_metrics.system_fingerprints)} fingerprints, {utils.format_size_si(batch_metrics.local_jsonl_size)} JSONL data, {(cached_tokens := batch_metrics.usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))} cached input + {batch_metrics.usage.get('prompt_tokens', 0) - cached_tokens} uncached input + {(reasoning_tokens := batch_metrics.usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0))} reasoning + {batch_metrics.usage.get('completion_tokens', 0) - reasoning_tokens} response = {batch_metrics.usage.get('total_tokens', 0)} total tokens, max {batch_metrics.max_seen_output_tokens} completion tokens, {self.S.metrics.batch.succeeded.completion_ratio:.3g} success completion ratio ({batch_metrics.completion_ratio:.3g} total), {batch_metrics.true_cost:.3f} true cost")
		log.info(f"{direct_metrics.num_requests} requests have been completed in DIRECT mode, entailing {len(direct_metrics.models)} models, {len(direct_metrics.system_fingerprints)} fingerprints, {utils.format_size_si(direct_metrics.local_jsonl_size)} JSONL data, {(cached_tokens := direct_metrics.usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))} cached input + {direct_metrics.usage.get('prompt_tokens', 0) - cached_tokens} uncached input + {(reasoning_tokens := direct_metrics.usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0))} reasoning + {direct_metrics.usage.get('completion_tokens', 0) - reasoning_tokens} response = {direct_metrics.usage.get('total_tokens', 0)} total tokens, max {direct_metrics.max_seen_output_tokens} completion tokens, {self.S.metrics.direct.succeeded.completion_ratio:.3g} success completion ratio ({direct_metrics.completion_ratio:.3g} total), {direct_metrics.true_cost:.3f} true cost")
		log.info(f"A total of {all_metrics.num_requests} requests have been COMPLETED, entailing {len(all_metrics.models)} models, {len(all_metrics.system_fingerprints)} fingerprints, {utils.format_size_si(all_metrics.local_jsonl_size)} JSONL data, {(cached_tokens := all_metrics.usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))} cached input + {all_metrics.usage.get('prompt_tokens', 0) - cached_tokens} uncached input + {(reasoning_tokens := all_metrics.usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0))} reasoning + {all_metrics.usage.get('completion_tokens', 0) - reasoning_tokens} response = {all_metrics.usage.get('total_tokens', 0)} total tokens, max {all_metrics.max_seen_output_tokens} completion tokens, {self.S.metrics.all.succeeded.completion_ratio:.3g} success completion ratio ({all_metrics.completion_ratio:.3g} total), {all_metrics.true_cost:.3f} true cost")

	# Log the push statistics after an update has occurred
	def log_push_stats(self):
		log.info(f"Session push statistics are now {self.session_push_stats.total_requests} requests, {self.session_push_stats.total_tokens_cost.input_tokens} input tokens, {self.session_push_stats.total_tokens_cost.cost:.3f} assumed cost")
		log.info(f"Task push statistics are now {self.S.task_push_stats.total_requests} requests, {self.S.task_push_stats.total_tokens_cost.input_tokens} input tokens, {self.S.task_push_stats.total_tokens_cost.cost:.3f} assumed cost")

	# Create a GPT request item
	def create_request_item(self, req: GPTRequest, parse_info: Optional[dict[str, Any]] = NONE) -> GPTRequestItem:
		# req = The request to wrap as a request item (req is copied if it needs to be modified)
		# parse_info = Custom explicit value for the parse information, regardless of whether auto-parse is enabled (useful when constructing a new GPTRequestItem based on a previously auto-parsed payload, as auto-parsing potentially edits a payload in such a way that it does not auto-parse as intended a second time)

		if parse_info is NONE:

			if self.auto_parse:

				parse_info = {}
				if self.endpoint == '/v1/chat/completions':  # Chat completions endpoint

					if req.payload.get('stream', False):
						raise ValueError("Streaming GPT requests are not supported")

					tools = req.payload.get('tools', openai.NOT_GIVEN)
					openai_parsing.validate_input_tools(tools=tools)

					response_format = req.payload.get('response_format', openai.NOT_GIVEN)
					resolved_response_format = openai_parsing.type_to_response_format_param(response_format=response_format)

					payload = req.payload.copy()
					if openai_utils.is_given(resolved_response_format):
						payload['response_format'] = resolved_response_format
					else:
						del payload['response_format']
					req = dataclasses.replace(req, payload=payload)

					if isinstance(response_format, type):
						parse_info['response_format_type'] = self.type_cache.cache_type(typ=response_format, verify=True)
					elif response_format is not resolved_response_format and (openai_utils.is_given(response_format) or openai_utils.is_given(resolved_response_format)):
						raise ValueError(f"Auto-parse unexpectedly changed the value of response_format: {response_format} vs {resolved_response_format}")

			else:
				parse_info = None

		self.max_request_id += 1
		return GPTRequestItem(id=self.max_request_id, req=req, parse_info=parse_info)

	# Generator: Make a number of sequential synchronous requests using the direct API (responses may return as multiple BatchResult instances if there are many requests)
	def direct_requests(self, reqs: Union[Iterable[Union[GPTRequest, GPTRequestItem, CachedGPTRequest]], GPTRequest, GPTRequestItem, CachedGPTRequest], yield_retry_reqs: bool) -> Iterable[tuple[Optional[utils.RevertStack], Optional[BatchResult], Optional[bool], Optional[list[CachedGPTRequest]]]]:
		# reqs = The requests to make using the direct API (raw, itemized and/or cached requests can be provided)
		# yield_retry_reqs = Whether to yield tuples (for complete transparency of operations) containing any retry requests that are generated and will be internally executed (in addition to, and generally alternating with, the tuples containing the virtual batch results for which the retry behavior can still be modified)
		# CAUTION: No raw GPTRequest instances should be provided that have already been edited by the auto-parse functionality at some point, and need to be paired with their existing parse_info (e.g. a GPTRequest extracted or derived from a GPTRequestItem or CachedGPTRequest) => In that case, use self.create_request_item(req=..., parse_info=...) instead and pass the resulting GPTRequestItem
		# The provided requests are auto-split into virtual batches based on direct request number/token limits and such
		# Either processes and returns results for all requests (plus associated retries) one virtual batch at a time, or eventually yields limited=True and discontinues (after a possible trailing retry requests yield), or raises an exception if an unresolvable issue occurs
		# If dry run is active, the returned results will have both resp_info=None and err_info=None
		# This method is implemented as a generator that returns tuples of:
		#   - A RevertStack that should be used to reversibly update and save the task state and output file(s)
		#   - The results for one virtual batch at a time (the total number of results obtained may exceed the number of requests given due to retries)
		#   - A boolean whether direct limits were reached in the process (meaning that at least one request could not be fully completed due to them, or only-process mode is active)
		#   - A list of retry requests generated based on the results of the batch, that will be internally executed in a further virtual batch
		# The possible combinations of yielded types are:
		#   - tuple[utils.RevertStack, BatchResult, bool, None] => The result of a virtual batch, ready to be processed (the retry behavior can still be modified by editing BatchResult while processing the results)
		#   - tuple[None, None, bool, None] => Status update whether the direct limits have been reached (generally when the bool is True but there is no BatchResult available that this update to the limited state can be yielded as part of)
		#   - tuple[utils.RevertStack, BatchResult, None, list[CachedGPTRequest]] => If yield_retry_reqs, a tuple containing a list of retry requests that were generated from the immediately previously yielded batch result
		# If the boolean is ever True that direct limits were reached, then no further batch results are sent (but one more set of retry requests may be sent, if some exist)
		# Note: The generator must have its close() method called even if an exception occurs => This is automatically handled by the Python interpreter when using a for-loop to iterate the generator, but is not guaranteed if manually using next() and such
		# Note: Calling close() prior to complete iteration or executing break or return statements while iterating in a for-loop causes a GeneratorExit exception to internally be raised, which makes the currently active RevertStack unwind with an exception and thereby revert all currently performed work!

		if self.only_process:
			log.warning("Only-process mode => Not making any direct requests")
			yield None, None, True, None  # Direct limits have immediately been reached if only-process mode
			return

		if isinstance(reqs, (GPTRequest, GPTRequestItem, CachedGPTRequest)):
			reqs_list = [reqs]
		else:
			reqs_list = list(reqs)

		num_input_reqs = len(reqs_list)
		log.info(f"Executing {num_input_reqs} requests using the direct API...")

		delayed_raise = utils.DelayedRaise()
		direct_limits_reached = False
		direct_limits_list = None
		num_direct_batches = 0
		num_total_errors = 0
		yielded_req_ids = []

		index = 0
		total_start_time = time.perf_counter()
		while index < len(reqs_list) and not direct_limits_reached:

			log.info('\xB7' * 60)
			delayed_raise.new_section()

			batch = BatchState()
			cached_reqs: dict[int, CachedGPTRequest] = {}

			post_batch_task_requests = self.S.task_push_stats.total_requests
			post_batch_session_requests = self.session_push_stats.total_requests
			post_batch_task_tokens_cost = self.S.task_push_stats.total_tokens_cost
			post_batch_session_tokens_cost = self.session_push_stats.total_tokens_cost

			while index < len(reqs_list) and not direct_limits_reached:

				cached_req = reqs_list[index]
				if isinstance(cached_req, GPTRequest):
					cached_req = self.create_request_item(req=cached_req)
				if isinstance(cached_req, GPTRequestItem):
					cached_req = CachedGPTRequest.from_item(item=cached_req, endpoint=self.endpoint, token_estimator=self.token_estimator, token_coster=self.token_coster)
				if not isinstance(cached_req, CachedGPTRequest):
					raise TypeError(f"Expected a CachedGPTRequest but got a {utils.get_class_str(type(cached_req))}")
				reqs_list[index] = cached_req

				next_task_requests = post_batch_task_requests + 1
				next_session_requests = post_batch_session_requests + 1
				next_task_tokens_cost = TokensCost(
					input_tokens=post_batch_task_tokens_cost.input_tokens + cached_req.info.tokens_cost.input_tokens,
					output_tokens=post_batch_task_tokens_cost.output_tokens + cached_req.info.tokens_cost.output_tokens,
					max_output_tokens=post_batch_task_tokens_cost.max_output_tokens + cached_req.info.tokens_cost.max_output_tokens,
					cost_input_direct=post_batch_task_tokens_cost.cost_input_direct + cached_req.info.tokens_cost.cost_input_direct,
					cost_input_batch=post_batch_task_tokens_cost.cost_input_batch,
					cost_output_direct=post_batch_task_tokens_cost.cost_output_direct + cached_req.info.tokens_cost.cost_output_direct,
					cost_output_batch=post_batch_task_tokens_cost.cost_output_batch,
				)
				next_session_tokens_cost = TokensCost(
					input_tokens=post_batch_session_tokens_cost.input_tokens + cached_req.info.tokens_cost.input_tokens,
					output_tokens=post_batch_session_tokens_cost.output_tokens + cached_req.info.tokens_cost.output_tokens,
					max_output_tokens=post_batch_session_tokens_cost.max_output_tokens + cached_req.info.tokens_cost.max_output_tokens,
					cost_input_direct=post_batch_session_tokens_cost.cost_input_direct + cached_req.info.tokens_cost.cost_input_direct,
					cost_input_batch=post_batch_session_tokens_cost.cost_input_batch,
					cost_output_direct=post_batch_session_tokens_cost.cost_output_direct + cached_req.info.tokens_cost.cost_output_direct,
					cost_output_batch=post_batch_session_tokens_cost.cost_output_batch,
				)

				next_num_requests = batch.num_requests + 1
				next_tokens_cost = batch.tokens_cost + cached_req.info.tokens_cost

				assert not batch.reasons and not direct_limits_reached

				if next_task_requests > self.max_task_requests:
					batch.reasons.append('Max task requests')
				if next_task_tokens_cost.input_tokens > self.max_task_tokens:
					batch.reasons.append('Max task tokens')
				if next_task_tokens_cost.cost > self.max_task_cost:
					batch.reasons.append('Max task cost')
				if next_session_requests > self.max_session_requests:
					batch.reasons.append('Max session requests')
				if next_session_tokens_cost.input_tokens > self.max_session_tokens:
					batch.reasons.append('Max session tokens')
				if next_session_tokens_cost.cost > self.max_session_cost:
					batch.reasons.append('Max session cost')
				if batch.reasons:
					direct_limits_reached = True
					direct_limits_list = sorted(batch.reasons)

				if next_num_requests > self.max_direct_requests:
					batch.reasons.append('Max direct requests')
				if next_tokens_cost.input_tokens > self.max_direct_tokens:
					batch.reasons.append('Max direct tokens')
				if next_tokens_cost.cost_direct > self.max_direct_cost:
					batch.reasons.append('Max direct cost')

				if batch.reasons:
					if batch.num_requests <= 0 and not direct_limits_reached:
						raise ValueError(f"The direct batch limits are too strict to allow even a single request to be added: Batch requests {next_num_requests} > {self.max_direct_requests} OR Batch tokens {next_tokens_cost.input_tokens} > {self.max_direct_tokens} OR Batch cost {next_tokens_cost.cost_direct} > {self.max_direct_cost}")
					batch.full_batch = True
					break

				req_id = cached_req.item.id
				batch.local_jsonl_size += cached_req.info.json_size  # noqa
				batch.num_requests = next_num_requests
				assert req_id not in batch.request_info and cached_req.info.tokens_cost.is_valid()
				batch.request_info[req_id] = cached_req.get_request_info()
				batch.tokens_cost = next_tokens_cost
				assert req_id not in cached_reqs
				cached_reqs[req_id] = cached_req

				post_batch_task_requests = next_task_requests
				post_batch_session_requests = next_session_requests
				post_batch_task_tokens_cost = next_task_tokens_cost
				post_batch_session_tokens_cost = next_session_tokens_cost

				index += 1

			if batch.num_requests <= 0 and direct_limits_reached:
				yield None, None, direct_limits_reached, None  # Direct limits have immediately been reached on attempting to create a direct batch
				break

			assert batch.num_requests == len(batch.request_info) > 0
			assert batch.tokens_cost.equals(TokensCost().add(req_info.tokens_cost for req_info in batch.request_info.values()))
			assert batch.request_info.keys() == cached_reqs.keys()

			num_direct_batches += 1
			self.max_direct_batch_id += 1
			batch.id = self.max_direct_batch_id
			batch.local_jsonl = f'direct-{batch.id}.jsonl'  # Note: Dummy file path that of course in general does not actually exist
			if not batch.reasons:
				batch.reasons.append('No more requests')
			log.info(f"Created direct batch {batch.id} = {'Full' if batch.full_batch else 'Trailing'} virtual batch of size {utils.format_size_si(batch.local_jsonl_size)} with {batch.num_requests} requests, {batch.tokens_cost.input_tokens} tokens, {batch.tokens_cost.cost_direct:.3f} assumed cost, due to reasons: {', '.join(sorted(batch.reasons))}")

			if self.dryrun:
				log.warning(f"{DRYRUN}Not actually executing requests in direct batch {batch.id}")
			else:
				log.info(f"Executing direct batch {batch.id}...")

			created_time = int(time.time())
			input_file_id = f'direct-file-input-{batch.id}'
			batch.remote = RemoteBatchState(
				file=openai.types.FileObject(
					id=input_file_id,
					bytes=batch.local_jsonl_size,
					created_at=created_time,
					filename=batch.local_jsonl,
					object='file',
					purpose='batch',
					status='processed',
				),
				batch=openai.types.Batch(
					id=f'direct_batch_{batch.id}',
					completion_window='',
					created_at=created_time,
					endpoint=self.endpoint,
					input_file_id=input_file_id,
					object='batch',
					status='in_progress',
					expires_at=created_time + 86400,
					in_progress_at=created_time,
					metadata=self.generate_batch_metadata(batch=batch),
				),
				finished=False,
			)

			raise_if_batch_failed = self.session_processed_failed_batches >= self.process_failed_batches >= 0
			batch_failed = False

			batch_start_time = time.perf_counter()
			printed_str = None
			printed_time = None
			verbosed_req_ids = set()

			def allow_other_print():
				nonlocal printed_str, printed_time
				utils.print_clear_line()
				printed_str = None
				printed_time = time.perf_counter()

			def safe_log_fn(msg: str):
				allow_other_print()
				log.error(msg)

			def print_verbose(req_id_: int, req_info_: RequestInfo, payload_: dict[str, Any], resp_info_: Optional[ResponseInfo], warn_infos_: list[ErrorInfo], err_info_: Optional[ErrorInfo]):
				if req_id_ not in verbosed_req_ids and (self.direct_verbose == DirectVerboseMode.ALWAYS or (self.direct_verbose >= DirectVerboseMode.WARN and warn_infos_) or (self.direct_verbose >= DirectVerboseMode.ERROR and err_info_ is not None)):
					verbose_printer = VerbosePrettyPrinter()
					verbose_msg = (
						"\u2193\u2193\u2193\n"
						f"{'\u2501' * verbose_printer.width}\n"
						f"\x1b[32mREQUEST ID:\x1b[0m {req_id_}\n"
						"\x1b[32mMETHOD:\x1b[0m POST\n"
						f"\x1b[32mENDPOINT:\x1b[0m {self.endpoint}\n"
						f"\x1b[32mREQUEST INFO:\x1b[0m\n{verbose_printer.pformat(req_info_)}\n"
						f"\x1b[32mREQUEST JSON:\x1b[0m\n{verbose_printer.pformat(payload_)}\n"
					)
					if resp_info_ is not None:
						verbose_msg += f"\x1b[32mRESPONSE INFO:\x1b[0m\n{verbose_printer.pformat(resp_info_)}\n"
					if warn_infos_:
						verbose_msg += f"\x1b[38;5;226mWARNING INFOS:\x1b[0m\n{verbose_printer.pformat(warn_infos_)}\n"
					if err_info_ is not None:
						verbose_msg += f"\x1b[38;5;196mERROR INFO:\x1b[0m\n{verbose_printer.pformat(err_info_)}\n"
					verbose_msg += ('\u2501' * verbose_printer.width)
					allow_other_print()
					log.info(verbose_msg)
					verbosed_req_ids.add(req_id_)

			result_info_map: dict[int, ResultInfo] = {}  # Maps: Request ID --> Result information
			warn_resp = utils.LogSummarizer(log_fn=log.warning, show_msgs=self.show_warnings)
			err_resp_retry, err_resp_fatal = [utils.LogSummarizer(log_fn=safe_log_fn, show_msgs=self.show_errors) for _ in range(2)]

			num_api_calls = 0
			num_batch_errors = 0
			for i, (req_id, req_info) in enumerate(batch.request_info.items(), 1):

				now = time.perf_counter()
				print_str = f"Direct batch {batch.id}: Executing request ID {req_id}... [Batch: {i}/{batch.num_requests} req, {num_batch_errors} err, {utils.format_duration(now - batch_start_time)}] [Total: {index - batch.num_requests + i}/{len(reqs_list)} req, {num_total_errors} err, {utils.format_duration(now - total_start_time)}] "
				if print_str != printed_str and (printed_time is None or now - printed_time >= self.remote_update_interval):
					utils.print_in_place(print_str)
					printed_str = print_str
					printed_time = now

				cached_req = cached_reqs[req_id]

				resp_info = None
				warn_infos = []

				if self.dryrun:
					err_info = None  # Note: The only instance where neither resp_info nor err_info is non-None (internal only - NOT yielded)

				else:

					try:

						resp_payload: Union[None, openai_chat.ChatCompletion, openai_chat.ParsedChatCompletion[object]]
						if self.endpoint == '/v1/chat/completions':  # Chat completions endpoint
							resp_payload = self.client.post(
								path='/chat/completions',
								body=openai_utils.maybe_transform(data=cached_req.item.req.payload, expected_type=openai_chat.CompletionCreateParams),
								options={},
								cast_to=openai_chat.ChatCompletion,
								stream=False,
								stream_cls=openai.Stream[openai_chat.ChatCompletionChunk],
							)
						else:
							raise EndpointError(f"Cannot post request to unrecognized endpoint: {self.endpoint}")
						num_api_calls += 1

					except openai.APIError as e:
						status_code = getattr(e, 'status_code', 200)
						if status_code == 200:
							err_info = ErrorInfo(fatal=True, type='PostError', subtype='APIError', msg=f"Post of request to endpoint '{self.endpoint}' failed with {utils.get_class_str(e)}: {e}")
						else:
							err_info = self.construct_status_code_err_info(status_code=status_code, response_error=e.body)

					except Exception as e:  # noqa
						if isinstance(e, EndpointError):
							raise
						else:
							err_info = ErrorInfo(fatal=True, type='PostError', subtype='Exception', msg=f"Post of request to endpoint '{self.endpoint}' failed with {utils.get_class_str(e)}: {e}")

					else:
						resp_payload, err_info, resp_model, resp_system_fingerprint, resp_usage = self.process_response_payload(req_info=req_info, req_payload=cached_req.item.req.payload, resp_payload=resp_payload, warn_infos=warn_infos, request_id=None)
						assert resp_model is not None and resp_system_fingerprint is not None and resp_usage is not None
						resp_info = ResponseInfo(payload=resp_payload, model=resp_model, system_fingerprint=resp_system_fingerprint, usage=resp_usage)

				print_verbose(req_id_=req_id, req_info_=req_info, payload_=cached_req.item.req.payload, resp_info_=resp_info, warn_infos_=warn_infos, err_info_=err_info)

				for warn_info in warn_infos:
					warn_resp.log(f"Direct batch {batch.id} request ID {req_id} got warning {warn_info.type}({warn_info.subtype}): {warn_info.msg}")

				if err_info is not None:
					err_msg = f"Direct batch {batch.id} request ID {req_id} got {'fatal' if err_info.fatal else 'retryable'} error {err_info.type}({err_info.subtype}): {err_info.msg}"
					if err_info.fatal:
						err_resp_fatal.log(err_msg)
						if raise_if_batch_failed:
							delayed_raise.add(msg=f"Request got fatal error {err_info.type}({err_info.subtype})")
						batch_failed = True
					else:
						err_resp_retry.log(err_msg)
					num_batch_errors += 1
					num_total_errors += 1

				assert resp_info is not None or err_info is not None or self.dryrun
				result_info_map[req_id] = self.update_result_retry(info=ResultInfo(
					req_id=req_id,
					req_jsonl_size=cached_req.info.json_size,
					req_payload=cached_req.item.req.payload,
					req_info=req_info,
					resp_info=resp_info,
					err_info=err_info,
					warn_infos=warn_infos,
					retry_counts=True,
					retry=False,
				))

			now = time.perf_counter()
			utils.print_in_place(f"Direct batch {batch.id}: Executed {batch.num_requests} batch requests with {num_batch_errors} errors in {utils.format_duration(now - batch_start_time)} [Total: {index}/{len(reqs_list)} req, {num_total_errors} err, {utils.format_duration(now - total_start_time)}]\n")

			warn_resp.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further warnings for direct batch {batch.id} (total {num_total} warnings)")
			err_resp_retry.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further retryable errors for direct batch {batch.id} (total {num_total} errors)")
			err_resp_fatal.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further fatal errors for direct batch {batch.id} (total {num_total} errors)")

			assert result_info_map.keys() == batch.request_info.keys()

			completed_time = int(time.time())
			batch.remote.status = 'completed'
			batch.remote.batch.completed_at = completed_time
			result_has_err_info = tuple(result_info.err_info is not None for result_info in result_info_map.values())
			if any(result_has_err_info):
				batch.remote.batch.error_file_id = f'direct-file-error-{batch.id}'
			batch.remote.batch.finalizing_at = completed_time
			if not all(result_has_err_info):
				batch.remote.batch.output_file_id = f'direct-file-output-{batch.id}'
			assert len(result_info_map) == batch.num_requests
			num_err_infos = sum(result_has_err_info)
			batch.remote.batch.request_counts = openai.types.BatchRequestCounts(
				completed=len(result_info_map) - num_err_infos,
				failed=num_err_infos,
				total=len(result_info_map),
			)
			batch.remote.finished = True

			log.info(f"Finished executing direct batch {batch.id}")

			result, batch_true_tokens_cost, batch_metrics_succeeded, batch_metrics_failed = self.process_result_info_map(
				batch=batch,
				batch_errors=[],
				result_info_map=result_info_map,
				req_payload_map={req_id: (cached_req.info.json_size, cached_req.item.req.payload) for req_id, cached_req in cached_reqs.items()},
				direct_mode=True,
			)

			if delayed_raise.have_section_errors():
				log.error(f"Aborting processing of direct batch {batch.id} results as errors were encountered (not updating saved state)")
				yield None, None, direct_limits_reached, None  # Direct limits may have been reached, so need to communicate that in the absence of passing/saving batch results
				continue
			elif batch_failed:
				log.warning(f"Processing direct batch {batch.id} results even though the direct batch at least partially failed")

			def revert_push_stats(task_requests: int, session_requests: int, task_tokens_cost: TokensCost, session_tokens_cost: TokensCost):
				self.S.task_push_stats.total_requests = task_requests
				self.session_push_stats.total_requests = session_requests
				self.S.task_push_stats.total_tokens_cost = task_tokens_cost
				self.session_push_stats.total_tokens_cost = session_tokens_cost

			def revert_metrics(api_metrics_direct: APIMetrics, api_metrics_all: APIMetrics):
				self.S.metrics.all = api_metrics_all
				self.S.metrics.direct = api_metrics_direct

			def revert_state(max_direct_batch_id: int, direct_history: list[BatchState], max_request_id: int):
				self.S.max_request_id = max_request_id
				self.S.direct_history[:] = direct_history
				self.S.max_direct_batch_id = max_direct_batch_id

			with utils.AtomicLogRevertStack(W=self.W) as rstack:

				rstack.callback(revert_push_stats, task_requests=self.S.task_push_stats.total_requests, session_requests=self.session_push_stats.total_requests, task_tokens_cost=self.S.task_push_stats.total_tokens_cost, session_tokens_cost=self.session_push_stats.total_tokens_cost)
				self.S.task_push_stats.total_requests = post_batch_task_requests
				self.session_push_stats.total_requests = post_batch_session_requests
				self.S.task_push_stats.total_tokens_cost = post_batch_task_tokens_cost
				self.session_push_stats.total_tokens_cost = post_batch_session_tokens_cost

				rstack.callback(revert_metrics, api_metrics_direct=copy.deepcopy(self.S.metrics.direct), api_metrics_all=copy.deepcopy(self.S.metrics.all))
				batch.true_tokens_cost = batch_true_tokens_cost
				batch.metrics = APIMetrics()
				for api_metrics in (batch.metrics, self.S.metrics.direct, self.S.metrics.all):
					api_metrics.num_api_calls += num_api_calls
					api_metrics.succeeded.add_metrics(batch_metrics_succeeded)
					api_metrics.failed.add_metrics(batch_metrics_failed)
					api_metrics.total.add_metrics(batch_metrics_succeeded, batch_metrics_failed)

				log.info(f"Performing task-specific processing of the direct batch {batch.id} results...")
				yielded_req_ids.extend(result.info.keys())
				yield rstack, result, direct_limits_reached, None  # Note: The task is permitted to update result.info[REQID].retry/retry_counts (both boolean) to reflect whether a request will be retried or not, and whether it counts towards the retry number (theoretically, even the payload or such could be tweaked to update the retry)

				for info in result.info.values():
					print_verbose(req_id_=info.req_id, req_info_=info.req_info, payload_=info.req_payload, resp_info_=info.resp_info, warn_infos_=info.warn_infos, err_info_=info.err_info)

				self.update_result_stats(result=result, batch_id=batch.id, direct_mode=True)
				self.log_result_summary(result=result, rstack=rstack)

				rstack.callback(revert_state, max_direct_batch_id=self.S.max_direct_batch_id, direct_history=self.S.direct_history.copy(), max_request_id=self.S.max_request_id)
				self.S.max_direct_batch_id = self.max_direct_batch_id
				batch.request_info = {}
				self.S.direct_history.append(batch)
				self.S.direct_history.sort()

				retry_reqs = [CachedGPTRequest.from_item(
					item=self.create_request_item(
						req=GPTRequest(
							payload=info.req_payload,
							meta=info.req_info.meta,
							retry_num=info.req_info.retry_num + 1 if info.retry_counts else info.req_info.retry_num,
						),
						parse_info=info.req_info.parse_info,
					),
					endpoint=self.endpoint,
					token_estimator=self.token_estimator,
					token_coster=self.token_coster,
				) for info in result.info.values() if info.retry]
				self.S.max_request_id = self.max_request_id

				if retry_reqs:
					log.info(f"{len(retry_reqs)} requests need to be retried and will be appended to the direct request queue...")
					reqs_list.extend(retry_reqs)

				if yield_retry_reqs:
					yield rstack, result, None, retry_reqs  # Note: No updates to result or retry_reqs are permitted!

				self.validate_state_queue(clean=False)
				self.save_state(rstack=rstack)

				if batch_failed:
					rstack.callback(setattr, self, 'session_processed_failed_batches', self.session_processed_failed_batches)
					self.session_processed_failed_batches += 1
					log.warning(f"Have processed a total of {self.session_processed_failed_batches} failed batches in this session")

		if index > 0:
			log.info('\xB7' * 60)
		if direct_limits_reached:
			log.warning(f"Direct limits were reached while executing requests using the direct API: {', '.join(direct_limits_list)}")
		log.info(f"Executed {min(index, num_input_reqs)} requests ({index} including retries) in {num_direct_batches} direct batches using the direct API")
		delayed_raise.raise_on_error(base_msg="Encountered errors while processing direct batches")
		if not direct_limits_reached:
			assert len(yielded_req_ids) == len(set(yielded_req_ids)) == index == len(reqs_list)
			assert all(isinstance(cached_req, CachedGPTRequest) for cached_req in reqs_list)
			assert yielded_req_ids == [cached_req.item.id for cached_req in reqs_list]
		if index > 0:
			self.log_push_stats()

	# Add a single request to the request pool without committing it to the request queue just yet (the request will be committed in the next call to commit_requests())
	def add_request(self, req: GPTRequest, parse_info: Optional[dict[str, Any]] = NONE) -> int:
		# req = The request to add to the request pool
		# parse_info = Optional custom explicit value for the parse information (refer to self.create_request_item())
		# Returns the associated unique request ID
		# Be careful not to inadvertently modify any part of req after adding the request (e.g. the meta/payload or any deeper part of it)
		item = self.create_request_item(req=req, parse_info=parse_info)
		cached_req = CachedGPTRequest.from_item(item=item, endpoint=self.endpoint, token_estimator=self.token_estimator, token_coster=self.token_coster)
		self.P.append(cached_req)
		return cached_req.item.id

	# Add multiple requests to the request pool without committing them to the request queue just yet (the requests will be committed in the next call to commit_requests())
	# Note that reqs can also for convenience be a generator that executes some loop over source samples and yields instances of GPTRequest
	def add_requests(self, reqs: Iterable[GPTRequest]) -> list[int]:
		# reqs = The requests to add to the request pool
		# Returns a list of the associated unique request IDs
		# Note: If custom explicit values for the parse information are required then use add_request() manually in a loop instead
		return [self.add_request(req) for req in reqs]

	# Context manager: Optionally add some request(s) to the request pool, then in any case commit all requests now in the pool to the request queue
	@contextlib.contextmanager
	def commit_requests(self, reqs: Union[Iterable[GPTRequest], GPTRequest, None] = None, rstack: Optional[utils.RevertStack] = None) -> ContextManager[tuple[utils.RevertStack, list[CachedGPTRequest]]]:
		# reqs = Optional requests to add to the request pool immediately prior to committing
		# rstack = Optional RevertStack to use instead of creating and entering a new one internally (a LogRevertStack to be precise, passing a RevertStack ensures the adding of requests is also reversible)
		# The context manager returns the currently active RevertStack and the list of committed cached GPT requests
		# The body of the context manager is executed within DelayKeyboardInterrupt and RevertStack context managers, the latter of which must completely reverse all actions taken if an exception is raised at any point whatsoever during the entered stack
		# The idea is that the body of the entered commit_requests() context manager should be used to save to disk whatever state is necessary so that all requests added to the GPT requester so far (given by a list[CachedGPTRequest]) do not get generated again in this or a new run (as they are now committed, i.e. locked in), even if the current run were to be immediately keyboard interrupted (or crash)

		if reqs is not None:

			if rstack is not None:
				def revert_pool(P: list[CachedGPTRequest]):
					self.P[:] = P
				rstack.callback(revert_pool, P=self.P.copy())

			if isinstance(reqs, GPTRequest):
				self.add_request(reqs)
			else:
				self.add_requests(reqs)

		def revert_queue(P: list[CachedGPTRequest], Q: list[Optional[CachedGPTRequest]]):
			self.P[:] = P
			self.Q[:] = Q

		def revert_queue_state(max_request_id: int, request_id_meta: dict[int, Optional[dict[str, Any]]]):
			self.S.max_request_id = max_request_id
			self.S.queue.request_id_meta = request_id_meta

		cm_rstack = utils.LogRevertStack(W=self.W) if rstack is None else contextlib.nullcontext(enter_result=rstack)
		with utils.DelayKeyboardInterrupt(), cm_rstack as rstack:
			cached_reqs = self.P.copy()
			if self.P or any(cached_req is None for cached_req in self.Q):
				rstack.callback(revert_queue, P=cached_reqs, Q=self.Q.copy())
				self.Q[:] = [cached_req for cached_req in self.Q if cached_req is not None]
				self.Q.extend(self.P)
				self.P.clear()
				rstack.callback(revert_queue_state, max_request_id=self.S.max_request_id, request_id_meta=self.S.queue.request_id_meta)
				self.S.max_request_id = self.max_request_id
				self.S.queue.request_id_meta = self.PQ.queue_request_id_meta()
				self.validate_state_queue(clean=True)
				self.queue.save(rstack=rstack)
				self.save_state(rstack=rstack)
			else:
				self.validate_state_queue(clean=True)
			yield rstack, cached_reqs  # Reaching this yield signifies that all requests that have been added to the request pool have been successfully committed to the request queue, queue file and state (BUT if the code executed during the yield raises an exception then ALL of this will be perfectly reversed)

	# Process the request queue and generate full batches as much as possible (also generate a trailing non-full batch with whatever requests are left if force=True)
	def batch_requests(self, force: bool = False) -> int:
		# force = Force all requests remaining at the end of forming batches to also be collected into a non-full batch
		# Returns the new number of unpushed batches
		# Note: Call can be skipped if not self.Q

		if self.only_process:
			log.warning("Only-process mode => Not creating any batches")
			return self.num_unpushed_batches()

		num_created = 0
		num_created_requests = 0

		index = 0
		batch = BatchState()
		while index < len(self.Q):

			num_unpushed_batches = self.num_unpushed_batches()
			if num_unpushed_batches >= self.max_unpushed_batches:
				log.info(f"Not batching {sum(cached_req is not None for cached_req in itertools.islice(self.Q, index, None))} requests left in the request queue as there are already {num_unpushed_batches} unpushed batches (max {self.max_unpushed_batches} allowed)")
				break

			batch_index = index
			batch_reqs = []
			batch = BatchState()

			num_nones = 0
			while index < len(self.Q):

				cached_req = self.Q[index]
				if cached_req is None:
					num_nones += 1
				else:

					next_local_jsonl_size = batch.local_jsonl_size + cached_req.info.json_size
					next_num_requests = batch.num_requests + 1
					next_tokens_cost = batch.tokens_cost + cached_req.info.tokens_cost

					assert not batch.reasons
					if next_num_requests > self.max_batch_requests:
						batch.reasons.append('Max batch requests')
					if next_local_jsonl_size > self.max_batch_size:
						batch.reasons.append('Max batch size')
					if next_tokens_cost.input_tokens > self.max_batch_tokens:
						batch.reasons.append('Max batch tokens')
					if next_tokens_cost.cost_batch > self.max_batch_cost:
						batch.reasons.append('Max batch cost')
					if batch.reasons:
						if batch.num_requests <= 0:
							raise ValueError(f"The batch limits are too strict to allow even a single request to be added: Batch requests {next_num_requests} > {self.max_batch_requests} OR Batch file size {next_local_jsonl_size} > {self.max_batch_size} OR Batch tokens {next_tokens_cost.input_tokens} > {self.max_batch_tokens} OR Batch cost {next_tokens_cost.cost_batch:.3f} > {self.max_batch_cost:.3f}")
						batch.full_batch = True
						break

					req_id = cached_req.item.id
					batch_reqs.append(cached_req.info.json)
					batch.local_jsonl_size = next_local_jsonl_size
					batch.num_requests = next_num_requests
					assert req_id not in batch.request_info and cached_req.info.tokens_cost.is_valid()
					batch.request_info[req_id] = cached_req.get_request_info()
					batch.tokens_cost = next_tokens_cost

				index += 1

			assert index - batch_index == batch.num_requests + num_nones
			assert batch.num_requests == len(batch.request_info)
			assert batch.tokens_cost.equals(TokensCost().add(req_info.tokens_cost for req_info in batch.request_info.values()))

			if (batch.full_batch or force) and batch.num_requests >= 1:

				if force:
					batch.reasons.append('Forced')

				def revert_state(max_batch_id: int, request_id_meta: dict[int, Optional[dict[str, Any]]], queue: list[Optional[CachedGPTRequest]]):
					self.Q[batch_index:index] = queue
					self.S.queue.request_id_meta = request_id_meta
					if self.S.batches and self.S.batches[-1] is batch:
						self.S.batches.pop()
					self.S.max_batch_id = max_batch_id

				with utils.AtomicLogRevertStack(W=self.W) as rstack:

					rstack.callback(revert_state, max_batch_id=self.S.max_batch_id, request_id_meta=self.S.queue.request_id_meta.copy(), queue=self.Q[batch_index:index])

					self.S.max_batch_id += 1
					batch.id = self.S.max_batch_id
					batch.local_jsonl = os.path.join(self.working_dir, f"{self.name_prefix}_batch{batch.id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl")

					self.S.batches.append(batch)  # Note: This is guaranteed to preserve ordering by ascending batch ID as the new batch ID is higher than any used so far
					for req_id, req_info in batch.request_info.items():
						assert self.S.queue.request_id_meta[req_id] == req_info.meta
						del self.S.queue.request_id_meta[req_id]
					self.Q[batch_index:index] = (None,) * (index - batch_index)

					if self.dryrun:
						log.warning(f"{DRYRUN}Did not create batch {batch.id} = {'Full' if batch.full_batch else 'Trailing'} local batch of size {utils.format_size_si(batch.local_jsonl_size)} with {batch.num_requests} requests, {batch.tokens_cost.input_tokens} tokens, {batch.tokens_cost.cost_batch:.3f} assumed cost [{os.path.basename(batch.local_jsonl)}] due to reasons: {', '.join(sorted(batch.reasons))}")
					else:
						with utils.SafeOpenForWrite(path=batch.local_jsonl, rstack=rstack) as file:
							file.writelines(batch_reqs)
							file_size = utils.get_file_size(file)
						assert file_size == batch.local_jsonl_size
						log.info(f"Created batch {batch.id} = {'Full' if batch.full_batch else 'Trailing'} local batch of size {utils.format_size_si(batch.local_jsonl_size)} with {batch.num_requests} requests, {batch.tokens_cost.input_tokens} tokens, {batch.tokens_cost.cost_batch:.3f} assumed cost [{os.path.basename(batch.local_jsonl)}] due to reasons: {', '.join(sorted(batch.reasons))}")

					self.validate_state_queue(clean=False)
					self.queue.save(rstack=rstack)
					self.save_state(rstack=rstack)

				num_created += 1
				num_created_requests += batch.num_requests

		if num_created > 0:
			log.info(f"Created {num_created} batches out of {num_created_requests} requests, leaving {self.PQ.queue_len} requests in the request queue for the future")
		else:
			log.info(f"The {batch.num_requests} available requests with {utils.format_size_si(batch.local_jsonl_size)}, {batch.tokens_cost.input_tokens} tokens, {batch.tokens_cost.cost_batch:.3f} assumed cost are currently not enough to trigger a batch")

		return self.num_unpushed_batches()

	# Load the local JSONL file of a batch
	def load_local_jsonl(self, batch: BatchState) -> dict[int, tuple[int, dict[str, Any]]]:
		# batch = Batch to load the local JSONL file of
		# Returns a map of Request ID --> (Request JSONL size, Request payload), whose keys are guaranteed to exactly match batch.request_info

		log.info(f"Loading batch {batch.id} input JSONL: {batch.local_jsonl}")

		req_payload_map: dict[int, tuple[int, dict[str, Any]]] = {}
		with open(batch.local_jsonl, 'rb') as file:
			file_size = utils.get_file_size(file)
			try:
				for line_bytes in file:
					req = json.loads(line_bytes.decode(encoding='utf-8'))
					if req['url'] != self.endpoint:
						raise ValueError(f"Endpoint mismatch: {req['url']} vs {self.endpoint}")
					req_id = int(re.fullmatch(r'id-([0-9]+)', req['custom_id']).group(1))
					if req_id in req_payload_map:
						raise ValueError(f"Duplicate request ID: {req_id}")
					req_payload_map[req_id] = (len(line_bytes), req['body'])
			except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError) as e:
				raise ValueError("Unexpected input JSONL data") from e

		total_line_size = sum(req_jsonl_size for req_jsonl_size, req_payload in req_payload_map.values())
		if len(req_payload_map) != batch.num_requests:
			raise ValueError(f"Input JSONL has unexpected number of requests: {len(req_payload_map)} vs {batch.num_requests}")
		elif req_payload_map.keys() != batch.request_info.keys():
			raise ValueError(f"Input JSONL does not contain exactly the expected request IDs, with disagreement in the keys: {sorted(req_payload_map.keys() ^ batch.request_info.keys())}")
		elif not file_size == total_line_size == batch.local_jsonl_size:
			raise ValueError(f"Input JSONL has unexpected file size: {file_size} vs {total_line_size} vs {batch.local_jsonl_size}")

		log.info(f"Loaded {batch.num_requests} requests of endpoint '{self.endpoint}' from batch {batch.id} input JSONL of size {utils.format_size_si(batch.local_jsonl_size)}")

		return req_payload_map

	# Check whether a batch is pushable (only makes sense if batch.remote is None)
	def batch_is_pushable(self, batch: BatchState) -> bool:
		# batch = The batch in question
		# Return whether the batch is pushable (True = Batch API) or unpushable (False = Direct API)
		return not self.force_direct and batch.num_requests >= self.min_batch_requests

	# Retrieve the number of unpushed local batches
	def num_unpushed_batches(self) -> int:
		# Returns the current number of unpushed local batches
		return sum(batch.remote is None for batch in self.S.batches)

	# Retrieve the number of unpushed local batches that are pushable
	def num_pushable_batches(self) -> int:
		# Returns the current number of unpushed local batches that are pushable
		return sum(batch.remote is None and self.batch_is_pushable(batch=batch) for batch in self.S.batches)

	# Retrieve the number of unpushed local batches that are unpushable
	def num_unpushable_batches(self) -> int:
		# Returns the current number of unpushed local batches that are unpushable
		return sum(batch.remote is None and not self.batch_is_pushable(batch=batch) for batch in self.S.batches)

	# Generate a string metadata dict to pass along with a batch to the remote
	def generate_batch_metadata(self, batch: BatchState) -> dict[str, str]:
		# batch = Batch to generate metadata for
		# Returns a pure-str key/value dict of metadata associated with the batch

		try:
			current_user = getpass.getuser()
		except OSError:
			current_user = str(os.getuid())

		return dict(
			host=os.uname().nodename,
			user=current_user,
			script=__file__,
			pid=str(os.getpid()),
			name_prefix=self.name_prefix,
			batch_id=str(batch.id),
			local_jsonl=batch.local_jsonl,
			state_file=self.state.path,
		)

	# Push as many pushable batches as possible for remote processing and return whether the batch pipeline is currently congested (i.e. a certain number of batches are complete and pending but cannot be pushed yet due to thresholds)
	def push_batches(self) -> bool:
		# Returns whether the batch pipeline is (now) congested
		# Note: Call can be skipped if self.num_unpushed_batches() <= 0

		if self.only_process:
			log.warning("Only-process mode => Not pushing any batches")
			return True  # Batch congestion

		remote_batches = 0
		remote_requests = 0
		remote_size = 0
		remote_tokens_cost = TokensCost()
		for batch in self.S.batches:
			if batch.remote is not None:
				remote_batches += 1
				remote_requests += batch.num_requests
				remote_size += batch.local_jsonl_size
				remote_tokens_cost += batch.tokens_cost

		num_pushed = 0
		reasons_nopush = set()
		for batch in self.S.batches:
			if batch.remote is None:

				next_task_requests = self.S.task_push_stats.total_requests + batch.num_requests
				next_session_requests = self.session_push_stats.total_requests + batch.num_requests
				next_task_tokens_cost = TokensCost(
					input_tokens=self.S.task_push_stats.total_tokens_cost.input_tokens + batch.tokens_cost.input_tokens,
					output_tokens=self.S.task_push_stats.total_tokens_cost.output_tokens + batch.tokens_cost.output_tokens,
					max_output_tokens=self.S.task_push_stats.total_tokens_cost.max_output_tokens + batch.tokens_cost.max_output_tokens,
					cost_input_direct=self.S.task_push_stats.total_tokens_cost.cost_input_direct,
					cost_input_batch=self.S.task_push_stats.total_tokens_cost.cost_input_batch + batch.tokens_cost.cost_input_batch,
					cost_output_direct=self.S.task_push_stats.total_tokens_cost.cost_output_direct,
					cost_output_batch=self.S.task_push_stats.total_tokens_cost.cost_output_batch + batch.tokens_cost.cost_output_batch,
				)
				next_session_tokens_cost = TokensCost(
					input_tokens=self.session_push_stats.total_tokens_cost.input_tokens + batch.tokens_cost.input_tokens,
					output_tokens=self.session_push_stats.total_tokens_cost.output_tokens + batch.tokens_cost.output_tokens,
					max_output_tokens=self.session_push_stats.total_tokens_cost.max_output_tokens + batch.tokens_cost.max_output_tokens,
					cost_input_direct=self.session_push_stats.total_tokens_cost.cost_input_direct,
					cost_input_batch=self.session_push_stats.total_tokens_cost.cost_input_batch + batch.tokens_cost.cost_input_batch,
					cost_output_direct=self.session_push_stats.total_tokens_cost.cost_output_direct,
					cost_output_batch=self.session_push_stats.total_tokens_cost.cost_output_batch + batch.tokens_cost.cost_output_batch,
				)

				next_remote_batches = remote_batches + 1
				next_remote_requests = remote_requests + batch.num_requests
				next_remote_size = remote_size + batch.local_jsonl_size
				next_remote_tokens_cost = remote_tokens_cost + batch.tokens_cost

				reasons = set()
				if not self.batch_is_pushable(batch=batch):
					reasons.add('Unpushable batch')
				if next_task_requests > self.max_task_requests:
					reasons.add('Max task requests')
				if next_task_tokens_cost.input_tokens > self.max_task_tokens:
					reasons.add('Max task tokens')
				if next_task_tokens_cost.cost > self.max_task_cost:
					reasons.add('Max task cost')
				if next_session_requests > self.max_session_requests:
					reasons.add('Max session requests')
				if next_session_tokens_cost.input_tokens > self.max_session_tokens:
					reasons.add('Max session tokens')
				if next_session_tokens_cost.cost > self.max_session_cost:
					reasons.add('Max session cost')
				if next_remote_batches > self.max_remote_batches:
					reasons.add('Max remote batches')
				if next_remote_requests > self.max_remote_requests:
					reasons.add('Max remote requests')
				if next_remote_size > self.max_remote_size:
					reasons.add('Max remote size')
				if next_remote_tokens_cost.input_tokens > self.max_remote_tokens:
					reasons.add('Max remote tokens')
				if next_remote_tokens_cost.cost_batch > self.max_remote_cost:
					reasons.add('Max remote cost')
				reasons_nopush.update(reasons)

				if not reasons:

					if self.dryrun:
						log.warning(f"{DRYRUN}Not pushing batch {batch.id} ({os.path.basename(batch.local_jsonl)}) of size {utils.format_size_si(batch.local_jsonl_size)}")
						reasons_nopush.add(DRYRUN)
					else:

						log.info(f"Pushing batch {batch.id} ({os.path.basename(batch.local_jsonl)}) of size {utils.format_size_si(batch.local_jsonl_size)}...")

						def revert_push_stats(task_requests: int, task_tokens_cost: TokensCost, session_requests: int, session_tokens_cost: TokensCost):
							self.S.task_push_stats.total_requests = task_requests
							self.S.task_push_stats.total_tokens_cost = task_tokens_cost
							self.session_push_stats.total_requests = session_requests
							self.session_push_stats.total_tokens_cost = session_tokens_cost

						with utils.AtomicLogRevertStack(W=self.W) as rstack:

							file_object = self.client.files.create(file=open(batch.local_jsonl, 'rb'), purpose='batch')
							rstack.callback(self.delete_remote_file, file_id=file_object.id)
							if file_object.bytes != batch.local_jsonl_size:
								log.warning(f"Uploaded input JSONL file '{file_object.id}' for batch {batch.id} has an unexpected size: {file_object.bytes} vs {batch.local_jsonl_size}")

							# noinspection PyTypeChecker
							batch_object = self.client.batches.create(
								completion_window='24h',
								endpoint=self.endpoint,
								input_file_id=file_object.id,
								metadata=self.generate_batch_metadata(batch=batch),
							)
							rstack.callback(self.cancel_remote_batch, batch_id=batch_object.id)

							rstack.callback(setattr, batch, 'remote', batch.remote)
							batch.remote = RemoteBatchState(file=file_object, batch=batch_object, finished=batch_object.status in FINISHED_BATCH_STATUSES)

							rstack.callback(revert_push_stats, task_requests=self.S.task_push_stats.total_requests, task_tokens_cost=self.S.task_push_stats.total_tokens_cost, session_requests=self.session_push_stats.total_requests, session_tokens_cost=self.session_push_stats.total_tokens_cost)
							self.S.task_push_stats.total_requests = next_task_requests
							self.S.task_push_stats.total_tokens_cost = next_task_tokens_cost
							self.session_push_stats.total_requests = next_session_requests
							self.session_push_stats.total_tokens_cost = next_session_tokens_cost

							self.validate_state_queue(clean=False)
							self.save_state(rstack=rstack)

							log.info(f"Pushed batch {batch.id} as '{batch.remote.batch.id}' based on '{batch.remote.file.id}' of size {utils.format_size_si(batch.remote.file.bytes)} with {batch.num_requests} requests, {batch.tokens_cost.input_tokens} tokens, {batch.tokens_cost.cost_batch:.3f} assumed cost (remote batch status: {batch.remote.batch.status})")

						remote_batches = next_remote_batches
						remote_requests = next_remote_requests
						remote_size = next_remote_size
						remote_tokens_cost = next_remote_tokens_cost

						num_pushed += 1

		num_unpushed_batches = self.num_unpushed_batches()
		num_pushable_batches = self.num_pushable_batches()
		num_unpushable_batches = self.num_unpushable_batches()
		num_unfinished_batches = self.num_unfinished_batches()
		num_finished_batches = self.num_finished_batches()
		assert num_pushable_batches + num_unpushable_batches == num_unpushed_batches
		assert num_unfinished_batches + num_finished_batches == remote_batches

		batch_congestion = (num_unpushed_batches >= self.max_unpushed_batches)  # Condition C = Whether the batch pipeline is congested (condition M is already guaranteed from pushing pushable batches above, so only need to check the second half of condition C)

		if reasons_nopush:
			log.info(f"Reasons encountered not to push certain batches right now: {', '.join(sorted(reasons_nopush))}")
		log.info(f"Pushed {num_pushed} batch(es) resulting in {num_unpushed_batches} unpushed local ({num_pushable_batches} pushable and {num_unpushable_batches} unpushable), {num_unfinished_batches} unfinished remote, and {num_finished_batches} finished remote batches{' [CONGESTED]' if batch_congestion else ''}")
		if num_pushed > 0:
			log.info(f"There are {remote_batches} remote batches with a total of {remote_requests} requests, {utils.format_size_si(remote_size)}, {remote_tokens_cost.input_tokens} tokens, {remote_tokens_cost.cost_batch:.3f} assumed cost")
			self.log_push_stats()

		return batch_congestion

	# Delete a local file (only log an error if deletion fails, never raise an exception)
	def delete_local_file(self, path: str):
		# path = Local file path to delete
		if self.dryrun:
			log.warning(f"{DRYRUN}Not deleting local file '{path}'")
		else:
			try:
				os.unlink(path)
			except Exception as e:  # noqa
				log.error(f"Failed to delete local file '{path}' due to {utils.get_class_str(e)}: {e}")

	# Delete a remote file (only log an error if deletion fails, never raise an exception)
	def delete_remote_file(self, file_id: str):
		# file_id = The remote file ID to delete
		if self.dryrun:
			log.warning(f"{DRYRUN}Not deleting remote file '{file_id}'")
		else:
			try:
				deleted_file = self.client.files.delete(file_id=file_id)
				assert deleted_file.id == file_id, "Remote file ID mismatch during deletion"
			except Exception as e:  # noqa
				log.error(f"Failed to delete remote file '{file_id}' due to {utils.get_class_str(e)}: {e}")

	# Cancel a remote batch (only log an error if cancellation fails, never raise an exception)
	def cancel_remote_batch(self, batch_id: str):
		# batch_id = The remote batch ID to cancel
		if self.dryrun:
			log.warning(f"{DRYRUN}Not canceling remote batch '{batch_id}'")
		else:
			try:
				cancelled_batch = self.client.batches.cancel(batch_id=batch_id)
				assert cancelled_batch.id == batch_id, "Remote batch ID mismatch during cancellation"
			except Exception as e:  # noqa
				log.error(f"Failed to cancel remote batch '{batch_id}' due to {utils.get_class_str(e)}: {e}")

	# Return how many remote batches there are
	def num_remote_batches(self) -> int:
		# Returns the current number of remote batches (whether finished or unfinished)
		return sum(batch.remote is not None for batch in self.S.batches)

	# Return how many unfinished remote batches there are according to the latest local state of information (see update_remote_batch_state())
	def num_unfinished_batches(self) -> int:
		# Returns the current number of unfinished remote batches
		return sum(batch.remote is not None and not batch.remote.finished for batch in self.S.batches)

	# Return how many finished remote batches there are according to the latest local state of information (see update_remote_batch_state())
	def num_finished_batches(self) -> int:
		# Returns the current number of finished remote batches
		return sum(batch.remote is not None and batch.remote.finished for batch in self.S.batches)

	# Update the current state of all pushed remote batches (unless dry run)
	def update_remote_batch_state(self, log_save: bool, last_save_time: Optional[float] = None) -> tuple[bool, int, Optional[float]]:
		# log_save = Whether to log if the state file is saved
		# last_save_time = If given, the last time the state was saved (as per time.perf_counter()) so that the wait save interval can be applied
		# Returns whether there are any finished remote batches, how many batch statuses were updated, and the new last save time (updated if state was saved)

		any_finished = False
		num_status_updates = 0
		status_changed = collections.defaultdict(list)
		for batch in self.S.batches:
			if batch.remote is not None:

				if batch.remote.finished:
					any_finished = True
				elif not self.dryrun:

					try:
						batch_status = self.client.batches.retrieve(batch_id=batch.remote.batch.id)
						if batch_status != batch.remote.batch:
							if batch_status.id != batch.remote.batch.id or batch_status.input_file_id != batch.remote.batch.input_file_id:
								raise ValueError(f"Remote returned incorrect IDs for query: {batch_status.id} != {batch.remote.batch.id} or {batch_status.input_file_id} != {batch.remote.batch.input_file_id}")
							if batch_status.status != batch.remote.batch.status:
								status_changed[batch_status.status].append(batch.id)
							batch.remote.batch = batch_status
							num_status_updates += 1
							if batch.remote.batch.status in FINISHED_BATCH_STATUSES:
								batch.remote.finished = True
								any_finished = True
					except (openai.OpenAIError, ValueError) as e:
						utils.print_clear_line()
						log.error(f"Failed to retrieve remote batch status with {utils.get_class_str(e)}: {e}")

		if num_status_updates != 0:
			now = time.perf_counter()
			save = (last_save_time is None or now - last_save_time >= self.wait_save_interval)
			if status_changed or (save and log_save):
				utils.print_clear_line()
			if status_changed:
				log.info(f"Detected change of remote batch status{'es' if sum(len(ids) for ids in status_changed.values()) > 1 else ''} to: {', '.join(f'{status} (batch{"es" if len(ids) > 1 else ""} {", ".join(str(idd) for idd in sorted(ids))})' for status, ids in status_changed.items())}")
			with utils.AtomicLogRevertStack(W=self.W) as rstack:
				self.validate_state_queue(clean=False)
				self.save_state(rstack=rstack, show_log=log_save, save=save)
				if save:
					last_save_time = now

		return any_finished, num_status_updates, last_save_time

	# Wait until there is at least one finished yet unprocessed remote batch or no more unfinished remote batches (unless dry run)
	def wait_for_batches(self):

		initial_num_unfinished_batches = self.num_unfinished_batches()
		if self.num_finished_batches() > 0 or initial_num_unfinished_batches <= 0 or self.dryrun:
			return

		start_time = time.perf_counter()
		num_status_updates = 0
		last_save_time = None
		printed_str = None

		while True:
			any_finished, new_status_updates, last_save_time = self.update_remote_batch_state(log_save=False, last_save_time=last_save_time)
			num_status_updates += new_status_updates
			num_unfinished_batches = self.num_unfinished_batches()
			if self.num_finished_batches() > 0 or num_unfinished_batches <= 0:
				utils.print_in_place(f"Waited {utils.format_duration(time.perf_counter() - start_time)} and {num_status_updates} status updates until {initial_num_unfinished_batches - num_unfinished_batches} batch(es) finished\n")
				return
			time.sleep((start_time - time.perf_counter()) % self.remote_update_interval)
			batches_in_progress = tuple(batch for batch in self.S.batches if batch.remote is not None and batch.remote.batch.request_counts is not None)
			next_batches = heapq.nlargest(n=3, iterable=((batch.id, batch.remote.batch.request_counts) for batch in batches_in_progress), key=lambda item: max(item[1].completed + item[1].failed, 0) / max(item[1].total, 1))
			next_batches.sort(key=operator.itemgetter(0))
			print_str = f"Still waiting on {num_unfinished_batches} unfinished batches after {utils.format_duration(time.perf_counter() - start_time)} and {num_status_updates} status updates{f' ({", ".join(f"batch {batch_id} = {counts.completed}+{counts.failed}/{counts.total}" for batch_id, counts in next_batches)})' if next_batches else ''}... "
			if print_str != printed_str:
				utils.print_in_place(print_str)
				printed_str = print_str

	# Generator: Execute, process and clean up after any unpushable batches
	def process_unpushable_batches(self) -> Iterable[tuple[Optional[utils.RevertStack], Optional[BatchResult], bool]]:
		# Either processes all unpushable batches one virtual (sub-)batch at a time, or eventually yields limited=True and discontinues, or raises an exception if an unresolvable issue occurs
		# If dry run is active, the returned results will have both resp_info=None and err_info=None
		# This method is implemented as a generator that returns tuples of:
		#   - A RevertStack that should be used to reversibly update and save the task state and output file(s)
		#   - The results for one direct/virtual batch at a time (unpushable batches are auto-split into potentially smaller direct batch(es))
		#   - A boolean whether direct limits were reached in the process (meaning that at least one request could not be fully completed due to them, or only-process mode is active)
		# The possible combinations of yielded types are:
		#   - tuple[utils.RevertStack, BatchResult, bool] => The result of a virtual batch, ready to be processed (the retry behavior can still be modified by editing BatchResult while processing the results)
		#   - tuple[None, None, bool] => Status update whether the direct limits have been reached (generally when the bool is True but there is no BatchResult available that this update to the limited state can be yielded as part of)
		# A single yield with no batch results and limited=True is expected if only-process mode is active
		# Note: The generator must have its close() method called even if an exception occurs => This is automatically handled by the Python interpreter when using a for-loop to iterate the generator, but is not guaranteed if manually using next() and such
		# Note: Calling close() prior to complete iteration or executing break or return statements while iterating in a for-loop causes a GeneratorExit exception to internally be raised, which makes the currently active RevertStack unwind with an exception and thereby revert all currently performed work!

		if self.only_process:
			log.warning("Only-process mode => Not executing and processing any unpushable batches")
			yield None, None, True  # Direct limits have immediately been reached if only-process mode
			return

		num_batches_finished = 0
		direct_limits_reached = False
		for batch in self.S.batches.copy():
			if batch.remote is None and not self.batch_is_pushable(batch=batch):

				log.info(f"Converting unpushable batch {batch.id} into direct batches for direct execution and processing...")
				batch_wiped = False

				if batch.request_info:

					req_payload_map = self.load_local_jsonl(batch=batch)

					cached_req_map: dict[int, CachedGPTRequest] = {}
					for req_id, req_info in batch.request_info.items():
						_, req_payload = req_payload_map[req_id]
						req_item = GPTRequestItem(id=req_id, req=GPTRequest(payload=req_payload, meta=req_info.meta, retry_num=req_info.retry_num), parse_info=req_info.parse_info)
						cached_req_map[req_id] = CachedGPTRequest.from_item(item=req_item, endpoint=self.endpoint, token_estimator=self.token_estimator, token_coster=self.token_coster)

					for rstack, result, limited, retry_reqs in self.direct_requests(reqs=cached_req_map.values(), yield_retry_reqs=True):  # Note: direct_requests() makes an internal list copy of the value of reqs, so it's okay to modify cached_req_map inside this loop

						assert not batch_wiped or result is None

						if retry_reqs is None:

							assert limited is not None and (rstack is None) == (result is None)

							yield rstack, result, limited  # Note: The task is permitted to update whether a request is retried or not

							if limited:
								direct_limits_reached = True

						else:

							assert rstack is not None and result is not None and limited is None

							def revert_batch(orig_batch: BatchState):
								for field in dataclasses.fields(BatchState):  # noqa
									setattr(batch, field.name, getattr(orig_batch, field.name))
							rstack.callback(revert_batch, orig_batch=copy.deepcopy(batch))

							for req_id, info in result.info.items():
								del cached_req_map[req_id]
								del batch.request_info[req_id]

							for cached_req in retry_reqs:
								req_id = cached_req.item.id
								assert req_id not in cached_req_map and req_id not in batch.request_info
								cached_req_map[req_id] = cached_req
								batch.request_info[req_id] = cached_req.get_request_info()

							assert cached_req_map.keys() == batch.request_info.keys()

							batch.local_jsonl_size = sum(cached_req.info.json_size for cached_req in cached_req_map.values())
							batch.num_requests = len(cached_req_map)
							batch.tokens_cost = TokensCost().add(cached_req.info.tokens_cost for cached_req in cached_req_map.values())
							if (result.info or retry_reqs) and 'Direct updated' not in batch.reasons:
								batch.full_batch = False
								batch.reasons.append('Direct updated')
							assert batch.true_tokens_cost is None and batch.metrics is None

							if batch.num_requests > 0:
								if self.dryrun:
									log.warning(f"{DRYRUN}Did not update batch {batch.id} = Unpushable local batch now of size {utils.format_size_si(batch.local_jsonl_size)} with {batch.num_requests} requests, {batch.tokens_cost.input_tokens} tokens, {batch.tokens_cost.cost_batch:.3f} assumed cost [{os.path.basename(batch.local_jsonl)}]")
								else:
									with utils.SafeOpenForWrite(path=batch.local_jsonl, rstack=rstack) as file:
										file.writelines(cached_req.info.json for cached_req in cached_req_map.values())
										file_size = utils.get_file_size(file)
									assert file_size == batch.local_jsonl_size
									log.info(f"Updated batch {batch.id} = Unpushable local batch now of size {utils.format_size_si(batch.local_jsonl_size)} with {batch.num_requests} requests, {batch.tokens_cost.input_tokens} tokens, {batch.tokens_cost.cost_batch:.3f} assumed cost [{os.path.basename(batch.local_jsonl)}]")
							else:
								rstack.callback(self.S.batches.__setitem__, slice(None), self.S.batches.copy())  # noqa
								self.S.batches.remove(batch)
								if self.dryrun:
									log.warning(f"{DRYRUN}Did not remove batch {batch.id} = Unpushable local batch that became empty [{os.path.basename(batch.local_jsonl)}]")
								else:
									utils.safe_unlink(path=batch.local_jsonl, rstack=rstack)
									log.info(f"Removed batch {batch.id} = Unpushable local batch that became empty [{os.path.basename(batch.local_jsonl)}]")
								num_batches_finished += 1
								batch_wiped = True

				else:
					with utils.AtomicLogRevertStack(W=self.W) as rstack:
						rstack.callback(self.S.batches.__setitem__, slice(None), self.S.batches.copy())  # noqa
						self.S.batches.remove(batch)
						if self.dryrun:
							log.warning(f"{DRYRUN}Did not remove {batch.id} = Unpushable local batch that was already empty [{os.path.basename(batch.local_jsonl)}]")
						else:
							utils.safe_unlink(path=batch.local_jsonl, rstack=rstack)
							log.info(f"Removed batch {batch.id} = Unpushable local batch that was already empty [{os.path.basename(batch.local_jsonl)}]")
						num_batches_finished += 1
						batch_wiped = True
						self.validate_state_queue(clean=False)
						self.save_state(rstack=rstack)

				if batch_wiped:
					log.info(f"Finished direct execution and processing of unpushable batch {batch.id}")
				else:
					assert direct_limits_reached
					log.warning(f"Stopping direct execution and processing of unfinished unpushable batch {batch.id} as direct limits have been reached")

				if direct_limits_reached:
					break

		log.info(f"Finished {num_batches_finished} unpushable batch(es) resulting in {self.num_unpushed_batches()} unpushed local batches ({self.num_pushable_batches()} pushable and {self.num_unpushable_batches()} unpushable)")

	# Update and prepare for the processing of remote batches
	def prepare_process_batches(self):
		self.update_remote_batch_state(log_save=True)  # Does not do much in case of dry run
		log.info(f"There are {self.num_unfinished_batches()} unfinished and {self.num_finished_batches()} finished REMOTE batches with a total of {sum(batch.num_requests for batch in self.S.batches if batch.remote is not None and not batch.remote.finished)} and {sum(batch.num_requests for batch in self.S.batches if batch.remote is not None and batch.remote.finished)} requests respectively")

	# Generator: Process and clean up after any finished remote batches (unless dry run)
	def process_batches(self) -> Iterable[tuple[utils.RevertStack, BatchResult]]:
		# This method is implemented as a generator that returns the results for one batch at a time in combination with a RevertStack, which should be used to reversibly update and save the task state and output file(s)
		# Note: The generator must have its close() method called even if an exception occurs => This is automatically handled by the Python interpreter when using a for-loop to iterate the generator, but is not guaranteed if manually using next() and such
		# Note: Calling close() prior to complete iteration or executing break or return statements while iterating in a for-loop causes a GeneratorExit exception to internally be raised, which makes the currently active RevertStack unwind with an exception and thereby revert all currently performed work!
		# Error handling comments guide for the implementation below:
		#   [NO ERROR] = The given situation is not innately an error
		#   [COVERED] = The given situation does not require explicit error consequences as other errors are guaranteed to trigger later anyway that cover this situation (if there is truly a problem)
		#   [DELAYED RAISE] = The fact that this situation occurred is noted, a corresponding exception will be raised later on, and the batch state will be kept as finished/unprocessed
		#   [RETRYABLE] = The given situation is retryable and thus only needs to be remembered as such

		self.prepare_process_batches()

		delayed_raise = utils.DelayedRaise()
		for batch in self.S.batches.copy():
			if batch.remote is not None and batch.remote.finished:

				log.info('\xB7' * 60)
				delayed_raise.new_section()

				if self.dryrun:
					log.warning(f"{DRYRUN}Not processing finished batch {batch.id} ({batch.remote.file.id}) containing {batch.num_requests} requests")
				else:

					log.info(f"Processing finished batch {batch.id} ({batch.remote.batch.id}) containing {batch.num_requests} requests...")

					raise_if_batch_failed = self.session_processed_failed_batches >= self.process_failed_batches >= 0
					batch_failed = False

					batch_errors: list[openai.types.BatchError] = batch.remote.batch.errors.data if batch.remote.batch.errors is not None and batch.remote.batch.errors.data is not None else []
					if batch.remote.batch.status == 'failed':
						log.error(f"Batch {batch.id} failed with errors:{''.join(f'\n    {batch_error}' for batch_error in batch_errors) if batch_errors else ' None'}")
						if raise_if_batch_failed:
							delayed_raise.add(msg=f"Batch status is '{batch.remote.batch.status}'")  # [DELAYED RAISE]
						batch_failed = True
					elif batch.remote.batch.status != 'completed' or batch_errors:
						# [COVERED] If this occurs then we just log an error, and continue trying to parse the batch anyway and deal with any errors that we encounter doing that, because that's the only thing that actually matters (e.g. expired and cancelled batches can still have valid results)
						log.log(logging.ERROR if batch_errors else logging.WARNING, f"Batch {batch.id} was overall unsuccessful with status '{batch.remote.batch.status}' and errors:{''.join(f'\n    {batch_error}' for batch_error in batch_errors) if batch_errors else ' None'}")

					req_payload_map = self.load_local_jsonl(batch=batch)

					result_info_map: dict[int, ResultInfo] = {}  # Maps: Request ID --> Result information
					warn_resp = utils.LogSummarizer(log_fn=log.warning, show_msgs=self.show_warnings)
					err_line_json, err_req_out_keys, err_req_id_fail, err_req_id_bad, err_req_id_dupl, err_resp_retry, err_resp_fatal = [utils.LogSummarizer(log_fn=log.error, show_msgs=self.show_errors) for _ in range(7)]

					for remote_file_id in (batch.remote.batch.error_file_id, batch.remote.batch.output_file_id):

						if remote_file_id is None:
							continue  # [NO ERROR] It is okay for e.g. an error file to not exist as long as then all requests are covered in the output file (or vice versa)

						try:
							file_content = self.client.files.content(file_id=remote_file_id)
						except openai.OpenAIError as e:
							log.error(f"Failed to retrieve remote file '{remote_file_id}' for batch {batch.id} with {utils.get_class_str(e)}: {e}")
							continue  # [COVERED] If the file contained requests, a request IDs mismatch will later occur (even if this is just a connectivity issue, there is not really much useful work that can be done at all without connectivity, so it's okay to error out)

						for line_num, line in enumerate(file_content.text.splitlines(), 1):

							if not line:
								continue  # [NO ERROR] Is actually not expected to occur, but skipping empty lines is harmless and no error in itself

							try:
								request_output = json.loads(line)
							except json.JSONDecodeError as e:
								err_line_json.log(f"Failed to parse line {line_num} to JSON in remote file '{remote_file_id}' for batch {batch.id} with {utils.get_class_str(e)}: {e}")
								continue  # [DELAYED RAISE]

							if 'custom_id' not in request_output or 'error' not in request_output or 'response' not in request_output:
								err_req_out_keys.log(f"Request output does not have the expected keys on line {line_num} of remote file '{remote_file_id}' for batch {batch.id}")
								continue  # [DELAYED RAISE]

							custom_id = request_output['custom_id']
							if isinstance(custom_id, str) and (match := re.fullmatch(r'id-([0-9]+)', custom_id)):
								req_id = int(match.group(1))
							else:
								err_req_id_fail.log(f"Failed to parse the request ID from line {line_num} of remote file '{remote_file_id}' for batch {batch.id}")
								continue  # [DELAYED RAISE]

							if req_id not in req_payload_map or req_id not in batch.request_info:
								err_req_id_bad.log(f"Unexpected request ID {req_id} in line {line_num} of remote file '{remote_file_id}' for batch {batch.id}")
								continue  # [DELAYED RAISE]
							req_jsonl_size, req_payload = req_payload_map[req_id]
							req_info = batch.request_info[req_id]

							if req_id in result_info_map:
								err_req_id_dupl.log(f"Encountered duplicate request ID {req_id} on line {line_num} of remote file '{remote_file_id}' for batch {batch.id}")
								continue  # [DELAYED RAISE]

							resp_info = None
							warn_infos = []

							if not (response := request_output['response']):
								err_info = ErrorInfo(fatal=True, type='ResponseField', subtype='MissingEmpty', msg='Response field is missing or empty')
							elif (status_code := response.get('status_code', None)) is None:
								err_info = ErrorInfo(fatal=True, type='StatusCode', subtype='MissingNull', msg='Response HTTP status code is missing or null')
							elif status_code != 200:
								err_info = self.construct_status_code_err_info(status_code=status_code, response_error=response.get('body', {}).get('error', None))
							elif not (resp_body := response.get('body', None)):
								err_info = ErrorInfo(fatal=True, type='ResponseBody', subtype='MissingEmpty', msg='Response body is missing or empty')
							else:

								resp_payload: Union[None, openai_chat.ChatCompletion, openai_chat.ParsedChatCompletion[object]] = None
								resp_model = resp_system_fingerprint = resp_usage = None

								if self.endpoint == '/v1/chat/completions':  # Chat completions endpoint

									if resp_body.get('object', None) != 'chat.completion':
										err_info = ErrorInfo(fatal=True, type='ResponseBody', subtype='ObjectIncorrect', data=resp_body.get('object', None), msg="Chat completion response body object field is not 'chat.completion'")
									else:

										try:
											# Converts response body: dict[str, Any] -> openai_chat.ChatCompletion
											resp_payload = openai_models.validate_type(type_=openai_chat.ChatCompletion, value=resp_body)
										except pydantic.ValidationError as e:
											err_info = ErrorInfo(fatal=True, type='ResponseBody', subtype='ValidationError', msg=f"Validation of chat completion response body as {utils.get_class_str(openai_chat.ChatCompletion)} model failed with {utils.get_class_str(e)}: {e}")
										else:
											resp_payload, err_info, resp_model, resp_system_fingerprint, resp_usage = self.process_response_payload(req_info=req_info, req_payload=req_payload, resp_payload=resp_payload, warn_infos=warn_infos, request_id=request_output.get('id', None))

								else:
									raise EndpointError(f"Cannot process response for unrecognized endpoint: {self.endpoint}")

								if resp_payload is not None:
									assert resp_model is not None and resp_system_fingerprint is not None and resp_usage is not None
									resp_info = ResponseInfo(payload=resp_payload, model=resp_model, system_fingerprint=resp_system_fingerprint, usage=resp_usage)

							request_error = request_output['error']
							if request_error is not None:
								# [COVERED] Any errors in parsing the response above are acceptable (and mostly expected) when there is a request error
								request_error_code = request_error['code']
								request_error_message = request_error['message']
								err_info = ErrorInfo(fatal=request_error_code not in RETRYABLE_ERROR_CODES, type='RequestError', subtype=request_error_code, data=request_error_message, msg=f"Request error {request_error_code}: {request_error_message}")

							for warn_info in warn_infos:
								warn_resp.log(f"Batch {batch.id} remote file '{remote_file_id}' line {line_num} request ID {req_id} got warning {warn_info.type}({warn_info.subtype}): {warn_info.msg}")

							if err_info is not None:
								err_msg = f"Batch {batch.id} remote file '{remote_file_id}' line {line_num} request ID {req_id} got {'fatal' if err_info.fatal else 'retryable'} error {err_info.type}({err_info.subtype}): {err_info.msg}"
								if err_info.fatal:
									err_resp_fatal.log(err_msg)
									if raise_if_batch_failed:
										delayed_raise.add(msg=f"Request got fatal error {err_info.type}({err_info.subtype})")  # [DELAYED RAISE]
									batch_failed = True
								else:
									err_resp_retry.log(err_msg)  # [RETRYABLE]

							assert resp_info is not None or err_info is not None
							result_info_map[req_id] = self.update_result_retry(info=ResultInfo(
								req_id=req_id,
								req_jsonl_size=req_jsonl_size,
								req_payload=req_payload,
								req_info=req_info,
								resp_info=resp_info,
								err_info=err_info,
								warn_infos=warn_infos,
								retry_counts=not (err_info is not None and err_info.type == 'RequestError' and err_info.subtype in ('batch_cancelled', 'batch_expired')),
								retry=False,
							))

					warn_resp.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further warnings for batch {batch.id} (total {num_total} warnings)")
					err_line_json.finalize(msg_fn=lambda num_omitted, num_total: f"Failed to parse a further {num_omitted} lines to JSON for batch {batch.id} (total {num_total} errors)")
					err_req_out_keys.finalize(msg_fn=lambda num_omitted, num_total: f"{num_omitted} further request outputs do not have the expected keys for batch {batch.id} (total {num_total} errors)")
					err_req_id_fail.finalize(msg_fn=lambda num_omitted, num_total: f"Failed to parse the request ID from {num_omitted} further lines for batch {batch.id} (total {num_total} errors)")
					err_req_id_bad.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further unexpected request IDs for batch {batch.id} (total {num_total} unexpected)")
					err_req_id_dupl.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further duplicate request IDs for batch {batch.id} (total {num_total} duplicates)")
					err_resp_retry.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further retryable errors for batch {batch.id} (total {num_total} errors)")
					err_resp_fatal.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further fatal errors for batch {batch.id} (total {num_total} errors)")

					if raise_if_batch_failed:
						delayed_raise.add(msg="Failed to parse output/error file line to JSON", count=err_line_json.num_msgs)
						delayed_raise.add(msg="Request output does not have the expected keys", count=err_req_out_keys.num_msgs)
						delayed_raise.add(msg="Failed to parse request ID from output/error file line", count=err_req_id_fail.num_msgs)
						delayed_raise.add(msg="Encountered an unexpected request ID", count=err_req_id_bad.num_msgs)
						delayed_raise.add(msg="Encountered a duplicate request ID", count=err_req_id_dupl.num_msgs)
					if err_line_json.num_msgs > 0 or err_req_out_keys.num_msgs > 0 or err_req_id_fail.num_msgs > 0 or err_req_id_bad.num_msgs > 0 or err_req_id_dupl.num_msgs > 0:
						batch_failed = True

					if result_info_map.keys() != batch.request_info.keys():
						log.error(f"Batch {batch.id} response does not contain exactly the expected request IDs, with disagreement in the keys: {sorted(result_info_map.keys() ^ batch.request_info.keys())}")
						if raise_if_batch_failed:
							delayed_raise.add(msg="Batch response does not contain exactly the expected request IDs")  # [DELAYED RAISE]
						batch_failed = True

					result, batch_true_tokens_cost, batch_metrics_succeeded, batch_metrics_failed = self.process_result_info_map(
						batch=batch,
						batch_errors=batch_errors,
						result_info_map=result_info_map,
						req_payload_map=req_payload_map,
						direct_mode=False,
					)

					if batch.remote.batch.status == 'cancelled' or result.stats.num_cancelled != 0:
						log.error(f"Batch {batch.id} response contains {result.stats.num_cancelled} cancelled requests")
						if raise_if_batch_failed:
							delayed_raise.add(msg="Batch response contains cancelled requests")  # [DELAYED RAISE]
						batch_failed = True

					if delayed_raise.have_section_errors():
						log.error(f"Aborting processing of finished batch {batch.id} ({batch.remote.batch.id}) as errors were encountered (leaving batch as it was)")
						continue  # [DELAYED RAISE]
					elif batch_failed:
						log.warning(f"Processing finished batch {batch.id} ({batch.remote.batch.id}) even though the batch at least partially failed")

					def revert_metrics(api_metrics_batch: APIMetrics, api_metrics_all: APIMetrics):
						self.S.metrics.all = api_metrics_all
						self.S.metrics.batch = api_metrics_batch
						batch.metrics = None
						batch.true_tokens_cost = None
					assert batch.true_tokens_cost is None and batch.metrics is None

					def revert_state(request_info: dict[int, RequestInfo], batches: list[BatchState], batch_history: list[BatchState], num_pass_failures: int):
						self.S.num_pass_failures = num_pass_failures
						self.S.batch_history[:] = batch_history
						self.S.batches[:] = batches
						batch.request_info = request_info

					with utils.DelayKeyboardInterrupt():

						with utils.LogRevertStack(W=self.W) as rstack:

							rstack.callback(revert_metrics, api_metrics_batch=copy.deepcopy(self.S.metrics.batch), api_metrics_all=copy.deepcopy(self.S.metrics.all))
							batch.true_tokens_cost = batch_true_tokens_cost
							batch.metrics = APIMetrics()
							for api_metrics in (batch.metrics, self.S.metrics.batch, self.S.metrics.all):
								api_metrics.num_api_calls += 1
								api_metrics.succeeded.add_metrics(batch_metrics_succeeded)
								api_metrics.failed.add_metrics(batch_metrics_failed)
								api_metrics.total.add_metrics(batch_metrics_succeeded, batch_metrics_failed)

							log.info(f"Performing task-specific processing of the batch {batch.id} results...")
							yield rstack, result  # Note: The task is permitted to update result.info[REQID].retry/retry_counts (both boolean) to reflect whether a request will be retried or not, and whether it counts towards the retry number (theoretically, even the payload or such could be tweaked to update the retry)

							self.update_result_stats(result=result, batch_id=batch.id, direct_mode=False)
							self.log_result_summary(result=result, rstack=rstack)

							rstack.callback(revert_state, request_info=batch.request_info, batches=self.S.batches.copy(), batch_history=self.S.batch_history.copy(), num_pass_failures=self.S.num_pass_failures)
							batch.request_info = {}
							self.S.batches.remove(batch)
							self.S.batch_history.append(batch)
							self.S.batch_history.sort()
							if not batch_failed:  # Note: Failed batches that are explicitly being allowed to clear out should not count towards pass failure, as otherwise clearing out failed batches would almost always lead to a pass failure, which is not the point
								if result.stats.num_pass <= 0 or result.stats.pass_ratio < self.min_pass_ratio:
									self.S.num_pass_failures += 1
								else:
									self.S.num_pass_failures = 0

							retry_infos = [info for info in result.info.values() if info.retry]
							if retry_infos:
								log.info(f"{len(retry_infos)} requests need to be retried and will be re-committed to the request queue...")
								for info in retry_infos:
									self.add_request(
										req=GPTRequest(
											payload=info.req_payload,
											meta=info.req_info.meta,
											retry_num=info.req_info.retry_num + 1 if info.retry_counts else info.req_info.retry_num,
										),
										parse_info=info.req_info.parse_info,
									)
								with self.commit_requests(rstack=rstack) as (_, cached_reqs):
									assert len(cached_reqs) >= len(retry_infos) > 0  # Note: There will always be something to commit and thus state validation and saving of the queue and state files will have happened internally
								if len(cached_reqs) == len(retry_infos):
									log.info(f"Committed {len(retry_infos)} retried requests to the request queue")
								else:
									log.info(f"Committed {len(cached_reqs) - len(retry_infos)} existing pooled requests and {len(retry_infos)} retried requests to the request queue")
							else:
								self.validate_state_queue(clean=False)
								self.save_state(rstack=rstack)

							if batch_failed:
								rstack.callback(setattr, self, 'session_processed_failed_batches', self.session_processed_failed_batches)
								self.session_processed_failed_batches += 1
								log.warning(f"Have processed a total of {self.session_processed_failed_batches} failed batches in this session")

						if batch.remote.batch.error_file_id is not None:
							self.delete_remote_file(file_id=batch.remote.batch.error_file_id)
						if batch.remote.batch.output_file_id is not None:
							self.delete_remote_file(file_id=batch.remote.batch.output_file_id)
						if batch.remote.file.id is not None:
							self.delete_remote_file(file_id=batch.remote.file.id)
						self.delete_local_file(path=batch.local_jsonl)

						log.info(f"Finished processing batch {batch.id} ({batch.remote.batch.id}) containing {batch.num_requests} requests")

					if self.S.num_pass_failures >= self.max_pass_failures:
						log.error(f"Have encountered {self.S.num_pass_failures} consecutive batch pass failures (max allowed {self.max_pass_failures})")
						delayed_raise.add(msg="Number of consecutive batch pass failures has reached limit")  # [DELAYED RAISE]

		delayed_raise.raise_on_error(base_msg="Encountered errors while processing finished remote batches")

	# Construct an ErrorInfo based on a HTTP status code (must be non-OK, i.e. not 200)
	@classmethod
	def construct_status_code_err_info(cls, status_code: int, response_error: Union[dict[str, Any], Any]) -> ErrorInfo:
		# status_code = Non-OK HTTP status code
		# response_error = Optional dictionary with details of the response error, potentially containing keys like message, type, code, param (anything other than a dict is ignored)
		# Returns the constructed ErrorInfo

		assert status_code != 200
		try:
			status_code_phrase = http.HTTPStatus(status_code).phrase
		except ValueError:
			status_code_phrase = 'Unknown Error'
		msg_parts = [f"Response HTTP {status_code} ({status_code_phrase})"]

		if isinstance(response_error, dict):
			if (response_error_type := response_error.get('type', None)) is not None:
				msg_parts.append(f' with error of type {response_error_type}')
				if (response_error_code := response_error.get('code', None)) is not None:
					msg_parts.append(f'.{response_error_code}')
				if (response_error_param := response_error.get('param', None)) is not None:
					msg_parts.append(f"(param='{response_error_param}')")
			if (response_error_message := response_error.get('message', None)) is not None:
				msg_parts.append(f': {'\\n'.join(format(response_error_message).splitlines())}')

		return ErrorInfo(fatal=status_code not in RETRYABLE_STATUS_CODES, type='StatusCode', subtype='NotOK', data=status_code, msg=''.join(msg_parts))

	# Process a response payload in an endpoint-specific manner (e.g. to apply auto-parsing)
	def process_response_payload(
		self,
		req_info: RequestInfo,                            # Request information
		req_payload: dict[str, Any],                      # Request payload
		resp_payload: Union[openai_chat.ChatCompletion],  # Validated response payload
		warn_infos: list[ErrorInfo],                      # List of warnings (mutable)
		request_id: Optional[str],                        # Request ID to write into the final response payload if the input response payload doesn't specify one (may edit the input response payload if no auto-parsing occurs)
	) -> tuple[Union[openai_chat.ChatCompletion, openai_chat.ParsedChatCompletion[object]], ErrorInfo, str, str, dict[str, Any]]:
		# Returns the final response payload (possibly untouched, possibly parsed), what error occurred (if any), the model used for the response, the system fingerprint of the system that generated the response (may be empty string if unknown), and the usage information as a possibly nested dict

		err_info = None
		if self.endpoint == '/v1/chat/completions':  # Chat completions endpoint

			request_id = getattr(resp_payload, '_request_id', request_id)

			if req_info.parse_info is not None:
				response_format = self.type_cache.retrieve_type(serial=req_info.parse_info['response_format_type']) if 'response_format_type' in req_info.parse_info else req_payload.get('response_format', openai.NOT_GIVEN)
				input_tools = req_payload.get('tools', openai.NOT_GIVEN)
				try:
					# Converts response payload: openai_chat.ChatCompletion -> openai_chat.ParsedChatCompletion[object]
					# The 'parsed' field of the choice messages CAN be None (e.g. if response_format is not given or a dict, if the raw choice message content is None, or if the message has a refusal)
					resp_payload = openai_parsing.parse_chat_completion(response_format=response_format, input_tools=input_tools, chat_completion=resp_payload)
				except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as e:
					err_info = ErrorInfo(fatal=False, type='AutoParse', subtype='FinishReason', msg=f"Auto-parsing of chat completion response as {utils.get_class_str(response_format) if openai_utils.is_given(response_format) else 'no'} format with {len(input_tools) if openai_utils.is_given(input_tools) else 'no'} tools failed with {utils.get_class_str(e)}: {e}")
				except (json.JSONDecodeError, pydantic.ValidationError) as e:
					err_info = ErrorInfo(fatal=False, type='AutoParse', subtype='ValidationError', msg=f"Auto-parsing of chat completion response as {utils.get_class_str(response_format) if openai_utils.is_given(response_format) else 'no'} format with {len(input_tools) if openai_utils.is_given(input_tools) else 'no'} tools failed with {utils.get_class_str(e)}: {e}")
				except Exception as e:  # noqa
					err_info = ErrorInfo(fatal=True, type='AutoParse', subtype='UnexpectedError', msg=f"Auto-parsing of chat completion response as {utils.get_class_str(response_format) if openai_utils.is_given(response_format) else 'no'} format with {len(input_tools) if openai_utils.is_given(input_tools) else 'no'} tools failed with {utils.get_class_str(e)}: {e}")
				else:
					if isinstance(response_format, type):
						if num_parsed_none := sum(choice.message.parsed is None for choice in resp_payload.choices):
							err_info = ErrorInfo(fatal=False, type='AutoParse', subtype='ParsedNone', msg=f"Auto-parsing of chat completion response as {utils.get_class_str(response_format)} format with {len(input_tools) if openai_utils.is_given(input_tools) else 'no'} tools yielded {num_parsed_none} missing parsed messages (e.g. due to refusals or missing content)")
						elif unexpected_parsed_types := collections.Counter(utils.get_class_str(type(choice.message.parsed)) for choice in resp_payload.choices if not isinstance(choice.message.parsed, response_format)):
							err_info = ErrorInfo(fatal=True, type='AutoParse', subtype='ParsedIncorrectType', msg=f"Auto-parsing of chat completion response as {utils.get_class_str(response_format)} format with {len(input_tools) if openai_utils.is_given(input_tools) else 'no'} tools yielded {unexpected_parsed_types.total()} parsed messages of incorrect type: {', '.join(sorted(unexpected_parsed_types.keys()))}")

			openai_models.add_request_id(obj=resp_payload, request_id=request_id)

			resp_model = resp_payload.model
			resp_system_fingerprint = resp_payload.system_fingerprint if resp_payload.system_fingerprint is not None else ''
			resp_usage = resp_payload.usage.to_dict(mode='json', use_api_names=False)

			if len(resp_payload.choices) != req_payload.get('n', 1):
				warn_infos.append(ErrorInfo(fatal=False, type='MessageChoices', subtype='IncorrectNum', msg=f"Expected {req_payload.get('n', 1)} response message choices but got {len(resp_payload.choices)}"))

			valid_choice = False
			for c, choice in enumerate(resp_payload.choices):
				if choice.finish_reason not in FINISH_REASONS_ALL:
					if err_info is None or not err_info.fatal:
						err_info = ErrorInfo(fatal=True, type='MessageChoice', subtype='FinishReason', data=c, msg=f"Choice {c} has the unrecognized finish reason '{choice.finish_reason}'")
				elif choice.finish_reason not in FINISH_REASONS_NOCONTENT and choice.message.content is None and choice.message.refusal is None and choice.message.audio is None:
					if err_info is None or not err_info.fatal:
						err_info = ErrorInfo(fatal=True, type='MessageChoice', subtype='NoContent', data=c, msg=f"Choice {c} has no content")
				elif choice.message.refusal is not None:
					warn_infos.append(ErrorInfo(fatal=False, type='MessageChoice', subtype='Refusal', data=c, msg=f"Choice {c} got refusal message: {choice.message.refusal}"))
				elif choice.finish_reason in FINISH_REASONS_FAILED:
					warn_infos.append(ErrorInfo(fatal=False, type='MessageChoice', subtype='FinishReason', data=c, msg=f"Choice {c} has the failed finish reason '{choice.finish_reason}'"))
				else:
					valid_choice = True
			if resp_payload.choices and not valid_choice and err_info is None:
				err_info = ErrorInfo(fatal=False, type='MessageChoices', subtype='NoneValid', msg="No valid response message choices")

		else:
			raise EndpointError(f"Cannot process response for unrecognized endpoint: {self.endpoint}")

		return resp_payload, err_info, resp_model, resp_system_fingerprint, resp_usage

	# Update a ResultInfo to reflect default error-dependent retry behavior (e.g. useful if info.err_info was modified)
	def update_result_retry(self, info: ResultInfo, retry_counts: Optional[bool] = None) -> ResultInfo:
		# info = The ResultInfo to update
		# retry_counts = Optionally override/set whether a possible retry would count towards the maximum number of allowed retries (None = Leave info.retry_counts untouched)
		# Returns the input ResultInfo (same instance, but modified)
		if retry_counts is not None:
			info.retry_counts = retry_counts
		info.retry = info.err_info is not None and (not info.err_info.fatal or self.retry_fatal_requests) and (info.req_info.retry_num < self.max_retries or not info.retry_counts)
		return info

	# Recalculate and update the result statistics of a BatchResult (intended to be used after task-specific processing)
	def update_result_stats(self, result: BatchResult, batch_id: int, direct_mode: bool) -> BatchResult:
		# result = The BatchResult to update
		# batch_id = The batch ID associated with the results
		# direct_mode = Whether the batch was executed using the direct API (True) or batch API (False)
		# Returns the input BatchResult (same instance, but modified)
		result.stats = self.calc_result_stats(result_infos=result.info.values())
		log.info(f"{'Direct batch' if direct_mode else 'Batch'} {batch_id} AFTER task-specific processing has {result.stats.num_requests} requests = {result.stats.num_success} success ({result.stats.num_empty} empty) + {result.stats.num_warned} warned + {result.stats.num_errored} errored ({result.stats.num_fatal} fatal, {result.stats.num_retryable} retryable, {result.stats.num_missing} missing, {result.stats.num_cancelled} cancelled, {result.stats.num_expired} expired) => {result.stats.num_pass} pass = {result.stats.pass_ratio:.1%} ratio")
		log.info(f"{'Direct batch' if direct_mode else 'Batch'} {batch_id} has {result.stats.num_final} final results ({result.stats.num_final_none} without issues + {result.stats.num_final_warn} with warnings + {result.stats.num_final_err} with errors), and needs {result.stats.num_retry} retries ({result.stats.num_retry_err} with errors + {result.stats.num_retry_warn} with warnings + {result.stats.num_retry_none} without issues, {result.stats.num_retry_counts} retries count)")
		return result

	# Calculate result statistics based on an iterable of ResultInfo's
	@classmethod
	def calc_result_stats(cls, result_infos: Iterable[ResultInfo]) -> ResultStats:
		# result_infos = Iterable of ResultInfo's
		# Returns the calculated result statistics

		num_requests = num_success = num_empty = num_warned = num_errored = num_retryable = num_cancelled = num_expired = num_fatal = num_missing = 0
		num_final = num_final_err = num_final_warn = num_final_none = num_retry = num_retry_err = num_retry_warn = num_retry_none = num_retry_counts = 0

		for info in result_infos:

			num_requests += 1

			if info.err_info is None:
				if info.resp_info is None:
					num_empty += 1
				if info.warn_infos:
					num_warned += 1
				else:
					num_success += 1
			else:
				num_errored += 1
				if info.err_info.fatal:
					num_fatal += 1
				else:
					num_retryable += 1
				if info.err_info.type == 'RequestError':
					assert not info.err_info.fatal
					if info.err_info.subtype == 'batch_cancelled':
						num_cancelled += 1
					elif info.err_info.subtype == 'batch_expired':
						num_expired += 1
				elif info.err_info.type == 'ResponseError':
					assert info.err_info.fatal
					if info.err_info.subtype == 'Missing':
						num_missing += 1

			if info.retry:
				num_retry += 1
				if info.err_info is not None:
					num_retry_err += 1
				elif info.warn_infos:
					num_retry_warn += 1
				else:
					num_retry_none += 1
				if info.retry_counts:
					num_retry_counts += 1
			else:
				num_final += 1
				if info.err_info is not None:
					num_final_err += 1
				elif info.warn_infos:
					num_final_warn += 1
				else:
					num_final_none += 1

		num_pass = num_success + num_expired
		result_stats = ResultStats(
			num_requests=num_requests,
			num_success=num_success,
			num_empty=num_empty,
			num_warned=num_warned,
			num_errored=num_errored,
			num_retryable=num_retryable,
			num_cancelled=num_cancelled,
			num_expired=num_expired,
			num_fatal=num_fatal,
			num_missing=num_missing,
			num_pass=num_pass,
			pass_ratio=num_pass / num_requests,
			num_final=num_final,
			num_final_err=num_final_err,
			num_final_warn=num_final_warn,
			num_final_none=num_final_none,
			num_retry=num_retry,
			num_retry_err=num_retry_err,
			num_retry_warn=num_retry_warn,
			num_retry_none=num_retry_none,
			num_retry_counts=num_retry_counts,
		)

		assert result_stats.num_requests == result_stats.num_success + result_stats.num_warned + result_stats.num_errored
		assert result_stats.num_errored == result_stats.num_retryable + result_stats.num_fatal
		assert result_stats.num_retryable >= result_stats.num_cancelled + result_stats.num_expired
		assert result_stats.num_fatal >= result_stats.num_missing
		assert result_stats.num_requests == result_stats.num_final + result_stats.num_retry
		assert result_stats.num_final == result_stats.num_final_err + result_stats.num_final_warn + result_stats.num_final_none
		assert result_stats.num_retry == result_stats.num_retry_err + result_stats.num_retry_warn + result_stats.num_retry_none
		assert result_stats.num_retry_counts <= result_stats.num_retry

		return result_stats

	# Process a ResultInfo map corresponding to the results of a batch into a BatchResult and associated information required for a state update
	def process_result_info_map(
		self,
		batch: BatchState,                                                # State of the batch
		batch_errors: list[openai.types.BatchError],                      # Batch-level errors associated with the batch
		result_info_map: dict[int, ResultInfo],                           # Result info map: Request ID --> Result information
		req_payload_map: dict[int, tuple[int, dict[str, Any]]],           # Request payload map: Request ID --> (Request JSONL size, Request payload)
		direct_mode: bool,                                                # Whether the batch was executed using the direct API (True) or batch API (False)
	) -> tuple[BatchResult, TokensCost, RequestMetrics, RequestMetrics]:  # Returns the overall batch result, the true tokens and cost, the metrics associated with succeeded requests, and the metrics associated with failed requests

		batch_models = collections.Counter()
		batch_system_fingerprints = collections.Counter()
		batch_predicted_input_tokens = 0
		batch_predicted_output_tokens = 0
		batch_max_output_tokens = 0
		batch_predicted_cost = 0.0
		batch_usage = {}
		batch_max_completion_tokens = 0
		for info in result_info_map.values():
			if info.resp_info is not None:
				batch_models[info.resp_info.model] += 1
				batch_system_fingerprints[info.resp_info.system_fingerprint] += 1
				RequestMetrics.usage_iadd(self_usage=batch_usage, other_usage=info.resp_info.usage)
				batch_max_completion_tokens = max(info.resp_info.usage.get('completion_tokens', 0), batch_max_completion_tokens)
				batch_predicted_input_tokens += info.req_info.tokens_cost.input_tokens
				batch_predicted_output_tokens += info.req_info.tokens_cost.output_tokens
				batch_max_output_tokens += info.req_info.tokens_cost.max_output_tokens
				batch_predicted_cost += info.req_info.tokens_cost.cost_direct if direct_mode else info.req_info.tokens_cost.cost_batch

		batch_input_tokens = batch_usage.get('prompt_tokens', 0)
		batch_cached_tokens = batch_usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
		batch_uncached_tokens = batch_input_tokens - batch_cached_tokens
		batch_completion_tokens = batch_usage.get('completion_tokens', 0)
		batch_reasoning_tokens = batch_usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
		batch_response_tokens = batch_completion_tokens - batch_reasoning_tokens
		batch_total_tokens = batch_usage.get('total_tokens', 0)
		batch_true_tokens_cost = TokensCost(
			input_tokens=batch_input_tokens,
			output_tokens=batch_completion_tokens,
			max_output_tokens=batch_max_output_tokens,
			cost_input_direct=self.token_coster.input_cost(direct=batch_input_tokens),
			cost_input_batch=self.token_coster.input_cost(batch=batch_input_tokens),
			cost_output_direct=self.token_coster.output_cost(direct=batch_completion_tokens),
			cost_output_batch=self.token_coster.output_cost(batch=batch_completion_tokens),
		)

		log.info(f"{'Direct batch' if direct_mode else 'Batch'} {batch.id} received {len(result_info_map)}/{batch.num_requests} responses, entailing {len(batch_models)} models, {len(batch_system_fingerprints)} fingerprints, {batch_input_tokens} input tokens ({batch_cached_tokens} cached and {batch_uncached_tokens} uncached, predicted {batch_predicted_input_tokens} input) + {batch_completion_tokens} completion tokens ({batch_reasoning_tokens} reasoning and {batch_response_tokens} response, max {batch_max_completion_tokens} tokens, completion ratio {batch_true_tokens_cost.completion_ratio:.3g}, assumed {batch_predicted_output_tokens} completion tokens) = {batch_total_tokens} total tokens, {batch_true_tokens_cost.cost_direct if direct_mode else batch_true_tokens_cost.cost_batch:.3f} true cost (predicted {batch_predicted_cost:.3f})")
		if (min_input_tokens := min(batch_input_tokens, batch_predicted_input_tokens)) != 0 and max(batch_input_tokens, batch_predicted_input_tokens) / min_input_tokens > self.warn_predicted_input_factor:
			log.warning(f"{'Direct batch' if direct_mode else 'Batch'} {batch.id} has an input token prediction mismatch in excess of a factor of {self.warn_predicted_input_factor:.3g}: {batch_input_tokens} true vs {batch_predicted_input_tokens} predicted tokens")
		if (min_completion_tokens := min(batch_completion_tokens, batch_predicted_output_tokens)) != 0 and max(batch_completion_tokens, batch_predicted_output_tokens) / min_completion_tokens > self.warn_assumed_completion_factor:
			log.warning(f"{'Direct batch' if direct_mode else 'Batch'} {batch.id} has an assumed completion token mismatch in excess of a factor of {self.warn_assumed_completion_factor:.3g}: {batch_completion_tokens} true vs {batch_predicted_output_tokens} assumed tokens / completion ratios are {batch_true_tokens_cost.completion_ratio:.3g} true vs {0.0 if batch_max_output_tokens == 0 else batch_predicted_output_tokens / batch_max_output_tokens} predicted vs {self.assumed_completion_ratio:.3g} configured")

		assert result_info_map.keys() <= batch.request_info.keys()
		for req_id, req_info in batch.request_info.items():
			if req_id not in result_info_map:
				req_jsonl_size, req_payload = req_payload_map[req_id]
				result_info_map[req_id] = self.update_result_retry(info=ResultInfo(
					req_id=req_id,
					req_jsonl_size=req_jsonl_size,
					req_payload=req_payload,
					req_info=req_info,
					resp_info=None,
					err_info=ErrorInfo(fatal=True, type='ResponseError', subtype='Missing', msg='Response is missing'),
					warn_infos=[],
					retry_counts=True,
					retry=False,
				))
		assert result_info_map.keys() == batch.request_info.keys()

		result_stats = self.calc_result_stats(result_infos=result_info_map.values())
		assert result_stats.num_requests == batch.num_requests == len(batch.request_info) == len(result_info_map)
		duration = max((at for at in (batch.remote.batch.failed_at, batch.remote.batch.completed_at, batch.remote.batch.expired_at, batch.remote.batch.cancelled_at) if at is not None), default=batch.remote.batch.created_at) - batch.remote.batch.created_at
		duration_hmin = utils.format_duration_hmin(duration)
		log.info(f"{'Direct batch' if direct_mode else 'Batch'} {batch.id} took {duration_hmin}{f' (max {batch.remote.batch.completion_window})' if batch.remote.batch.completion_window else ''} for {result_stats.num_requests} requests = {result_stats.num_success} success ({result_stats.num_empty} empty) + {result_stats.num_warned} warned + {result_stats.num_errored} errored ({result_stats.num_fatal} fatal, {result_stats.num_retryable} retryable, {result_stats.num_missing} missing, {result_stats.num_cancelled} cancelled, {result_stats.num_expired} expired) => {result_stats.num_pass} pass = {result_stats.pass_ratio:.1%} ratio (PRIOR to task-specific processing)")

		batch_metrics_succeeded = RequestMetrics()
		batch_metrics_failed = RequestMetrics()
		for info in result_info_map.values():
			batch_metrics = (batch_metrics_succeeded if info.resp_info is not None and info.err_info is None and not info.warn_infos else batch_metrics_failed)
			batch_metrics.num_requests += 1
			batch_metrics.local_jsonl_size += info.req_jsonl_size
			if info.resp_info is not None:  # Note: By definition always true for succeeded requests (unless dry run is active)
				batch_metrics.models[info.resp_info.model] += 1
				batch_metrics.system_fingerprints[info.resp_info.system_fingerprint] += 1
				RequestMetrics.usage_iadd(self_usage=batch_metrics.usage, other_usage=info.resp_info.usage)
				batch_metrics.max_output_tokens += info.req_info.tokens_cost.max_output_tokens
				num_prompt_tokens = info.resp_info.usage.get('prompt_tokens', 0)
				num_cached_tokens = info.resp_info.usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
				num_uncached_tokens = num_prompt_tokens - batch_cached_tokens
				num_completion_tokens = info.resp_info.usage.get('completion_tokens', 0)
				batch_metrics.max_seen_output_tokens = max(num_completion_tokens, batch_metrics.max_seen_output_tokens)
				if direct_mode:
					batch_metrics.true_cost += self.token_coster.cost(input_direct=num_uncached_tokens, input_cached=num_cached_tokens, output_direct=num_completion_tokens)
				else:
					batch_metrics.true_cost += self.token_coster.cost(input_batch=num_prompt_tokens, output_batch=num_completion_tokens)

		result = BatchResult(
			batch=batch,
			duration=duration,
			duration_hmin=duration_hmin,
			errors=batch_errors,
			info=dict(sorted(result_info_map.items())),
			stats=result_stats,
		)
		assert all(req_id == info.req_id and not (info.err_info is None and info.retry) for req_id, info in result.info.items())

		return result, batch_true_tokens_cost, batch_metrics_succeeded, batch_metrics_failed

	@classmethod
	def log_result_summary(cls, result: BatchResult, rstack: utils.RevertStack):
		# result = BatchResult to log a summary of
		# rstack = Possible LogRevertStack (subclass of RevertStack) to use for logging
		if isinstance(rstack, utils.LogRevertStack):
			rstack.log({
				'Result/batch_id': result.batch.id,
				'Result/direct_batch': result.batch.remote is None or result.batch.remote.batch.id.startswith('direct_batch_'),
				'Result/duration_hours': result.duration / 3600,
				'Result/num_batch_errors': len(result.errors),
				**{f'Result/{key}': value for key, value in dataclasses.asdict(result.stats).items()},  # noqa
			})
# EOF
