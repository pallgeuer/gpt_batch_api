# Logging configuration

# Imports
import logging

# Constants
LOGGER_NAME = 'gpt_batch_api'

# Configure and return the default logger
def configure_default_logger() -> logging.Logger:
	log_ = logging.getLogger(LOGGER_NAME)
	log_.setLevel(logging.INFO)
	return log_

# Configure logger
log = configure_default_logger()
# EOF
