# OpenAI GPT batch API
#
# Note:
#  - The environment variable OPENAI_API_KEY is required for any OpenAI API requests to work
#  - The current remote files can be manually managed at: https://platform.openai.com/storage
#  - The current remote batches can be manually managed at: https://platform.openai.com/batches

# TODO: Organise and structure this file (e.g. utils.py)

# Imports
import os
import logging
import contextlib
import dataclasses
from typing import Optional, Iterable, Self, Any, Protocol, ContextManager
import tqdm
import requests
import wandb
import openai
from . import utils
from .logger import log
from . import gpt_requester

# Protocol-based type annotation for configuration parameters (e.g. argparse.Namespace, omegaconf.DictConfig)
class Config(Protocol):

	def __getattr__(self, name: str) -> Any:
		...

	def __getitem__(self, key: str) -> Any:
		...

# TODO: Argparse suboptions (as namespace that can be passed to one of these classes) -> Don't need as hydra (have hydra suboptions?)? But good for others?
def foo(cfg: Config):
	log.info(f"Hey there! {cfg.foo}")
	requester = gpt_requester.GPTRequester(working_dir='outputs/_gpt_batch_api/gpt_batch_api', name_prefix='prefix')
	with requester:
		log.info("ENTERED")
	log.info("EXITED")
# EOF
