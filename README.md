# GPT Batch API Python Library

**Author:** Philipp Allgeuer

**Version:** 1.0.3

A Python library for efficiently interacting with OpenAI's GPT API in batch mode. This library helps handle multiple requests in a single batch to streamline and optimize API usage, making it ideal for high-volume non-real-time text and image processing applications. The Batch API provides a significantly faster and more cost-effective solution than performing multiple single API requests individually.

This library deals with all the complexities involved with having a safe, robust, cost-controlled, and restartable Large Language Model (LLM) batch processing application. This includes error handling, strict atomic control of state, and robustness to SIGINTs (keyboard interrupts, i.e. `Ctrl+C`) and crashes. Isolated calls to the standard non-batch (i.e. direct) API are also supported to allow efficient individual low-volume runs, or to more efficiently finish off the last few pending requests.

The library supports `wandb` integration to allow for the graphical/remote monitoring of progress of long-running processing applications.

## Getting Started

Environment setup instructions and additional useful run commands are provided in `commands.txt`. See below for a condensed quickstart guide using pip to conveniently install `gpt_batch_api`.

Applications should subclass the `TaskManager` class to implement the desired batch LLM task (see demos in `task_manager_demo.py` as well as the documentation in the `TaskManager` source code). Note that _very_ complex tasks could benefit from directly interacting with the underlying `GPTRequester` class (refer to how this class is used in `TaskManager`), but this should rarely be required.

**Useful environment variables (with example values):**
- `OPENAI_API_KEY`: [**Required**] The API key for authenticating requests to the OpenAI API (e.g. `sk-...`)
- `OPENAI_ORG_ID`: The organization ID associated with the OpenAI account, if required (e.g. `org-...`)
- `OPENAI_PROJECT_ID`: An identifier for the specific OpenAI project, used for tracking usage and billing (e.g. `proj_...`)
- `OPENAI_BASE_URL`: The base URL of where to direct API requests (e.g. `https://api.openai.com/v1`)
- `OPENAI_ENDPOINT`: The default endpoint to use for `GPTRequester` instances, if not explicitly otherwise specified on a per-`GPTRequester` basis (e.g. `/v1/chat/completions`)
- `WANDB_API_KEY`: [**Required if wandb support is enabled**] The API key for authenticating with Weights & Biases (e.g. `ff63...`, obtained from https://wandb.ai/authorize)

**Useful links:**
- Manage the defined OpenAI projects: https://platform.openai.com/settings/organization/projects
- View the OpenAI API rate and usage limits (and usage tier): https://platform.openai.com/settings/organization/limits
- Monitor the OpenAI API usage (costs, credits and bills): https://platform.openai.com/settings/organization/usage
- Manually monitor/manage the stored files on the OpenAI server: https://platform.openai.com/storage
- Manually monitor/manage the started batches on the OpenAI server: https://platform.openai.com/batches

## Quickstart

The `gpt_batch_api` library is available on PyPi, allowing you to quickly get started. Start by creating a virtual Python environment (e.g. `conda` or `venv`):
```bash
conda create -n gpt_batch_api python=3.12
conda activate gpt_batch_api
# OR...
python -m venv gpt_batch_api  # <-- Must be Python 3.12+
source gpt_batch_api/bin/activate
```
Then (after potentially installing some dependencies via conda if desired, see `commands.txt`) install `gpt_batch_api` (PyPi Python package built using [gpt_batch_api_build](https://github.com/pallgeuer/gpt_batch_api_build)):
```bash
pip install gpt_batch_api
```
If planning to use the `wandb` support (it is enabled by default), then ensure `wandb` is logged in (only required once):
```bash
wandb login
```
We can verify in an interactive `python` that the `gpt_batch_api` library has successfully been installed:
```python
import gpt_batch_api
print(gpt_batch_api.__version__)                              # Version
print(gpt_batch_api.TaskManager, gpt_batch_api.GPTRequester)  # Two main library classes
import os
print(os.path.join(os.path.dirname(gpt_batch_api.__file__), 'commands.txt'))  # Location of the installed commands.txt file (refer to this for command/script help)
```
Verify that the `gpt_batch_api` scripts can be run:
```bash
python -m gpt_batch_api.task_manager_demo --help
python -m gpt_batch_api.wandb_configure_view --help
```
Test running a script that actually makes API calls (requires less than 0.01 USD):
```bash
export OPENAI_API_KEY=sk-...  # <-- Set the OpenAI API key
export WANDB_API_KEY=...      # <-- Set the wandb API key (a project called gpt_batch_api is created/used and can be used to monitor the following run in real-time)
python -m gpt_batch_api.task_manager_demo --task_dir /tmp/gpt_batch_api_tasks --task utterance_emotion --model gpt-4o-mini-2024-07-18 --cost_input_direct_mtoken 0.150 --cost_input_cached_mtoken 0.075 --cost_input_batch_mtoken 0.075 --cost_output_direct_mtoken 0.600 --cost_output_batch_mtoken 0.300 --min_batch_requests 200 --max_direct_requests 40  # <-- The last two arguments avoid actually using the Batch API (as this can take a while to complete, and this is just a quick test)
python -m gpt_batch_api.wandb_configure_view --dst_entity ENTITY  # <-- [Substitute correct ENTITY! / Only need to execute this once ever per project!] Then go to https://wandb.ai/ENTITY/gpt_batch_api and select the saved view called 'GPT Batch API', and then click 'Copy to my workspace'
# Output files: Refer to /tmp/gpt_batch_api_tasks directory
# Note: If task_dir is not specified then a tasks directory will be auto-created inside the installed site-packages location, which is probably not desired in general
```
Now you are ready to implement your own custom tasks, and make full robust use of the power of the Batch API!

## Implementing a Custom Task

In order to define and run your own task, **refer to the example task implementations in `task_manager_demo.py`**, including the `main()` function and how the tasks are run.

The general steps to creating your own tasks are:

1) Read the documentation comments at the beginning of the `TaskManager` class, which outline which methods should be overridden, what sources of command line arguments are possible, and what simple properties the design of the task state, task output, and request metadata format need to satisfy.
2) Design/plan (e.g. on paper) the task-specific data format of the task state, task output, and request metadata format, so that all properties are satisfied.
3) If structured outputs are to be used in the requests, define a `pydantic.BaseModel` for the JSON schema that the LLM responses should strictly adhere to.
4) Decide on a task output file class (e.g. `DataclassOutputFile`, `DataclassListOutputFile`, or a custom `TaskOutputFile` implementation) and possibly define an appropriate subclass (e.g. to specify `Dataclass`), and define any associated dataclasses, pydantic models, and such.
5) Implement a custom task-specific subclass of `TaskManager`, given all the information from the previous steps. The subclass often needs to load data from file as input for generating the required requests and completing the task. This can be implemented inside the subclass, or can be implemented in separate code that e.g. then just passes pre-loaded data or a data loader class to the __init__ method of the `TaskManager` subclass. Refer to the documentation within each of the methods to override in the `TaskManager` class source code.
6) Ensure Python logging is configured appropriately (e.g. see `utils.configure_logging()`, or otherwise use `utils.ColorFormatter` manually to help ensure that warnings and errors stand out in terms of color).
7) Use `argparse` or `hydra` to configure command line arguments and pass them to the custom `TaskManager` subclass on init (refer to `main()` in `task_manager_demo.py`, and `config/gpt_batch_api.yaml`).
8) Run the custom task manager by constructing an instance of the class and calling `run()`.
