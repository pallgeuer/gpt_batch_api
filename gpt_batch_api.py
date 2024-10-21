# OpenAI GPT batch API
#
# Note:
#  - The environment variable OPENAI_API_KEY is required for any OpenAI API requests to work
#  - The current remote files can be manually managed at: https://platform.openai.com/storage
#  - The current remote batches can be manually managed at: https://platform.openai.com/batches

# Imports
import logging
import tqdm
import filelock
import requests
import wandb
import openai
from logger import log

# Logging configuration
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# TODO: Argparse suboptions (as namespace that can be passed to one of these classes)
def foo():
	log.info("Hey there!")
# EOF
