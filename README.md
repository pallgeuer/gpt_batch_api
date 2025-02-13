# GPT Batch API Python Library

**Author:** Philipp Allgeuer

**Version:** 1.0.0

A Python library for efficiently interacting with OpenAI's GPT API in batch mode. This library helps handle multiple requests in a single batch to streamline and optimize API usage, making it ideal for high-volume non-real-time text and image processing applications. The Batch API provides a significantly faster and more cost-effective solution than performing multiple single API requests individually.

This library deals with all the complexities involved with having a safe, robust, cost-controlled, and restartable Large Language Model (LLM) batch processing application. This includes error handling, strict atomic control of state, and robustness to SIGINTs (keyboard interrupts, i.e. `Ctrl+C`) and crashes. Isolated calls to the standard non-batch (i.e. direct) API are also supported to allow efficient individual low-volume runs, or to more efficiently finish off the last few pending requests.

The library supports `wandb` integration to allow for the graphical/remote monitoring of progress of long-running processing applications.

## Getting Started

Environment setup instructions and additional useful run commands are provided in `commands.txt`.

Applications should subclass the `TaskManager` class to implement the desired batch LLM task (see demos in `task_manager_demo.py` as well as the documentation in the `TaskManager` source code). Note that _very_ complex tasks could benefit from directly interacting with the underlying `GPTRequester` class (refer to how this class is used in `TaskManager`), but this should rarely be required.

**Useful environment variables (with example values):**
- `OPENAI_API_KEY`: [**Required**] The API key for authenticating requests to the OpenAI API (e.g. `sk-...`)
- `OPENAI_ORG_ID`: The organization ID associated with the OpenAI account, if required (e.g. `org-...`)
- `OPENAI_PROJECT_ID`: An identifier for the specific OpenAI project, used for tracking usage and billing (e.g. `proj_...`)
- `OPENAI_BASE_URL`: The base URL of where to direct API requests (e.g. `https://api.openai.com/v1`)
- `OPENAI_ENDPOINT`: The default endpoint to use for `GPTRequester` instances, if not explicitly otherwise specified on a per-`GPTRequester` basis (e.g. `/v1/chat/completions`)
- `WANDB_API_KEY`: [**Required if Wandb support is enabled**] The API key for authenticating with Weights & Biases (e.g. `ff63...`)

**Useful links:**
- Manage the defined OpenAI projects: https://platform.openai.com/settings/organization/projects
- View the OpenAI API rate and usage limits (and usage tier): https://platform.openai.com/settings/organization/limits
- Monitor the OpenAI API usage (costs, credits and bills): https://platform.openai.com/settings/organization/usage
- Manually monitor/manage the stored files on the OpenAI server: https://platform.openai.com/storage
- Manually monitor/manage the started batches on the OpenAI server: https://platform.openai.com/batches

## Implementing a Custom Task

In order to define and run your own task, refer to the example task implementations in `task_manager_demo.py`, including the `main()` function and how the tasks are run. The general steps to creating your own tasks are:

1) Read the documentation comments at the beginning of the `TaskManager` class, which outline which methods should be overridden, what sources of command line arguments are possible, and what simple properties the design of the task state, task output, and request metadata format need to satisfy.
2) Design/plan (e.g. on paper) the task-specific data format of the task state, task output, and request metadata format, so that all properties are satisfied.
3) If structured outputs are to be used in the requests, define a `pydantic.BaseModel` for the JSON schema that the LLM responses should strictly adhere to.
4) Decide on a task output file class (e.g. `DataclassOutputFile`, `DataclassListOutputFile`, or a custom `TaskOutputFile` implementation) and possibly define an appropriate subclass (e.g. to specify `Dataclass`), and define any associated dataclasses, pydantic models, and such.
5) Implement a custom task-specific subclass of `TaskManager`, given all the information from the previous steps. The subclass often needs to load data from file as input for generating the required requests and completing the task. This can be implemented inside the subclass, or can be implemented in separate code that e.g. then just passes pre-loaded data or a data loader class to the __init__ method of the `TaskManager` subclass. Refer to the documentation within each of the methods to override in the `TaskManager` class source code.
6) Ensure Python logging is configured appropriately (e.g. see `utils.configure_logging()`, it is also helpful if warnings and errors stand out in terms of color, see `utils.ColorFormatter`).
7) Use `argparse` or `hydra` to configure command line arguments and pass them to the custom `TaskManager` subclass on init (refer to `main()` in `task_manager_demo.py`, and `config/gpt_batch_api.yaml`).
8) Run the custom task manager by constructing an instance of the class and calling `run()`.
