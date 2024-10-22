# OpenAI GPT batch API
#
# Note:
#  - The environment variable OPENAI_API_KEY is required for any OpenAI API requests to work
#  - The current remote files can be manually managed at: https://platform.openai.com/storage
#  - The current remote batches can be manually managed at: https://platform.openai.com/batches

# Imports
import math
import time
import logging
from typing import Optional, Iterable, Self, Any, Protocol, ContextManager
import tqdm
import filelock
import requests
import wandb
import openai
from . import utils
from .logger import log

# Logging configuration
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# TODO: Argparse suboptions (as namespace that can be passed to one of these classes)
def foo():
	log.info("Hey there!")

# Lock file class
class LockFile:

	def __init__(self, lock_file: str, timeout: float = -1, poll_interval: float = 0.25, status_interval: float = 5.0):
		self.lock = filelock.FileLock(lock_file=lock_file)
		self.timeout = timeout
		self.poll_interval = poll_interval
		if self.poll_interval < 0.01:
			raise ValueError(f"Poll interval must be at least 0.01s: {self.poll_interval}")
		self.status_interval = status_interval
		if self.status_interval < 0.01:
			raise ValueError(f"Status interval must be at least 0.01s: {self.status_interval}")

	def __enter__(self) -> Self:
		start_time = time.perf_counter()
		print_time = start_time + self.status_interval
		if self.timeout >= 0:
			timeout_time = start_time + self.timeout
			timeout_str = f'/{utils.format_duration_unit(self.timeout)}'
		else:
			timeout_time = math.inf
			timeout_str = ''
		now = start_time
		while True:
			try:
				self.lock.acquire(timeout=max(min(timeout_time, print_time) - now, 0), poll_interval=self.poll_interval)
				utils.print_clear_line()
				return self
			except filelock.Timeout:
				now = time.perf_counter()
				if now >= timeout_time:
					utils.print_in_place(f"Failed to acquire lock in {utils.format_duration_unit(now - start_time)}: {self.lock.lock_file}\n")
					raise
				elif now >= print_time:
					utils.print_in_place(f"Still waiting on lock after {utils.format_duration_unit(now - start_time)}{timeout_str}: {self.lock.lock_file} ")
					print_time += self.status_interval

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		self.lock.release()
		return False

# EOF
