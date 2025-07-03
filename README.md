# GPT Batch API Python Library

**Author:** Philipp Allgeuer

**Version:** 1.0.4

A Python library for efficiently interacting with OpenAI's [Batch API](https://platform.openai.com/docs/guides/batch). This library helps handle multiple requests in a single batch to streamline and optimize API usage, making it ideal for high-volume non-real-time text and image processing applications. The Batch API provides a significantly faster and more cost-effective solution than performing multiple single API requests individually using the [direct API](https://platform.openai.com/docs/api-reference/chat).

This library deals with all the complexities involved with having a safe, robust, cost-controlled, and restartable Large Language Model (LLM) batch processing application. This includes error handling, strict atomic control of state, and robustness to SIGINTs (keyboard interrupts, i.e. `Ctrl+C`) and crashes. Isolated calls to the standard direct (i.e. non-batch) API are also supported to allow efficient individual low-volume runs, or to more efficiently finish off the last few pending requests of an otherwise batched run. Locally hosted LLMs (e.g. via [Ollama](https://ollama.com)) can also be used instead of OpenAI's models, if inference is restricted to the direct API (see [Local Inference](#local-inference)).

The library supports `wandb` integration to allow for the graphical/remote monitoring of progress of long-running processing applications.

## Getting Started

Refer to [Installation](#installation) below for instructions to conveniently install `gpt_batch_api` via pip, and [Task Manager Demo](#task-manager-demo) for instructions to run a demo of the library.

Applications that wish to use this library need to implement code that defines the task at hand, i.e. code that generates the required LLM requests, as well as code that processes the corresponding responses of the LLM. This is done by subclassing the `TaskManager` class to implement the desired batch LLM task (refer to the demos in [task_manager_demo.py](task_manager_demo.py), as well as the extensive documentation in the `TaskManager` [source code](task_manager.py)). Refer to [Implementing a Custom Task](#implementing-a-custom-task) for more information. Note that _very_ complex tasks could benefit from directly interacting with the underlying `GPTRequester` class (refer to how this class is used in `TaskManager`), but this should rarely be required in practice.

## Installation

The `gpt_batch_api` library is [available on PyPi](https://pypi.org/project/gpt-batch-api), allowing you to quickly get started. Start by creating a virtual Python environment (e.g. via `conda` or `venv`):
```bash
conda create -n gpt_batch_api python=3.12
conda activate gpt_batch_api
# OR...
python -m venv gpt_batch_api  # <-- Must be Python 3.12+
source gpt_batch_api/bin/activate
```
Then install `gpt_batch_api` ([PyPi Python package](https://pypi.org/project/gpt-batch-api) built using [gpt_batch_api_build](https://github.com/pallgeuer/gpt_batch_api_build)):
```bash
pip install gpt_batch_api
```
Alternatively, you can clone this source repository and just install the requirements:
```bash
pip install -r requirements.txt  # OR: requirements-dev.txt
```
In this case, to make sure Python finds the cloned source repository you can either `cd /path/to/gpt_batch_api/..` (i.e. change to the parent directory), or use `PYTHONPATH`.

If you plan to use the `wandb` support (recommended - it is enabled by default unless you specify `--no_wandb`), then ensure `wandb` is logged in (only required once ever):
```bash
wandb login
```
We can verify in an interactive `python` that the `gpt_batch_api` library has successfully been installed:
```python
import gpt_batch_api
print(gpt_batch_api.__version__)                              # Version
print(gpt_batch_api.TaskManager, gpt_batch_api.GPTRequester)  # Two main library classes
```
Verify that the `gpt_batch_api` scripts can be run:
```bash
python -m gpt_batch_api.task_manager_demo --help
python -m gpt_batch_api.wandb_configure_view --help
```
Test running a script that actually makes API calls, requiring less than 0.01 USD (refer to [Useful Hints](#useful-hints) for the exported environment variables):
```bash
export OPENAI_API_KEY=sk-...  # <-- Set the OpenAI API key
export WANDB_API_KEY=...      # <-- Set the wandb API key (a project called gpt_batch_api is created/used and can be used to monitor the following run in real-time)
python -m gpt_batch_api.task_manager_demo --task_dir /tmp/gpt_batch_api_tasks --task utterance_emotion --model gpt-4o-mini-2024-07-18 --cost_input_direct_mtoken 0.150 --cost_input_cached_mtoken 0.075 --cost_input_batch_mtoken 0.075 --cost_output_direct_mtoken 0.600 --cost_output_batch_mtoken 0.300 --force_direct  # <-- The last argument actually avoids use of the Batch API and forces use of the direct API instead (as the batch API can take a while to complete and this command should just be a quick test)
xdg-open /tmp/gpt_batch_api_tasks/utterance_emotion_output.jsonl  # <-- Opens the task-specific output file generated by the previous command
python -m gpt_batch_api.wandb_configure_view --dst_entity ENTITY  # <-- [Substitute correct ENTITY! / Only need to execute this once ever per project!] Then go to https://wandb.ai/ENTITY/gpt_batch_api and select the saved view called 'GPT Batch API', and then click 'Copy to my workspace'
```
The `task_manager_demo.py` script places its output files in the `--task_dir` directory, in this case `/tmp/gpt_batch_api_tasks`. Note that if `--task_dir` is not specified then a tasks directory will be auto-created inside the installed site-packages location, which is probably not desired in general.

Now you are ready to implement your own [custom tasks](#implementing-a-custom-task), and make full robust use of the power of the Batch API!

## Run Commands

Here is a general model configuration that is used for the commands in this section (you can update it to suit):
```bash
MODELARGS=(--model gpt-4o-mini-2024-07-18 --cost_input_direct_mtoken 0.150 --cost_input_cached_mtoken 0.075 --cost_input_batch_mtoken 0.075 --cost_output_direct_mtoken 0.600 --cost_output_batch_mtoken 0.300)
```

### Task Manager Demo

- **Command:**
  ```bash
  export OPENAI_API_KEY=sk-...  # <-- Set the OpenAI API key
  export WANDB_API_KEY=...      # <-- Set the wandb API key (a project called gpt_batch_api is created/used and can be used to monitor the following run in real-time)
  python -m gpt_batch_api.task_manager_demo --help
  python -m gpt_batch_api.task_manager_demo --task char_codes "${MODELARGS[@]}"
  python -m gpt_batch_api.task_manager_demo --task utterance_emotion "${MODELARGS[@]}"
  ```
  Refer to [Useful Hints](#useful-hints) for the exported environment variables.
- **Arguments:**
  - Refer to `--help`.
  - If you wish to change the directory in which files are created to store the ongoing and final state, and output data of the command (recommended if installed via pip), use:
    ```bash
    --task_dir /path/to/dir
    ```
  - If you wish not to use wandb for logging and monitoring, use:
    ```bash
    --no_wandb
    ```
- **Outputs:**
  - `gpt_batch_api/tasks/char_codes_*`
  - `gpt_batch_api/tasks/utterance_emotion_*`
  - Associated wandb run in web browser (useful for monitoring progress)
- **Verify:**
  - Verify that the final data output file(s) contain reasonable and complete data (e.g. `gpt_batch_api/tasks/char_codes_output*`, `gpt_batch_api/tasks/utterance_emotion_output*`)

### Useful Hints

**Useful environment variables:**
- `WANDB_API_KEY`: [**Required if wandb support is enabled**] The API key for authenticating with Weights & Biases (e.g. `ff63...`, obtained from https://wandb.ai/authorize)
- `OPENAI_API_KEY`: [**Required**] The API key for authenticating requests to the OpenAI API (e.g. `sk-...`, see also `--openai_api_key`)
- `OPENAI_ORG_ID`: The organization ID associated with the OpenAI account, if required (e.g. `org-...`, see also `--openai_organization`)
- `OPENAI_PROJECT_ID`: An identifier for the specific OpenAI project, used for tracking usage and billing (e.g. `proj_...`, see also `--openai_project`)
- `OPENAI_BASE_URL`: The base URL of where to direct API requests (e.g. `https://api.openai.com/v1` or `http://IP:PORT/v1`, see also `--client_base_url`)
- `OPENAI_ENDPOINT`: The default endpoint to use for `GPTRequester` instances, if not explicitly otherwise specified on a per-`GPTRequester` basis (e.g. `/v1/chat/completions`, see also `--chat_endpoint`)

**Useful links:**
- Manage the defined OpenAI projects: https://platform.openai.com/settings/organization/projects
- View the OpenAI API rate and usage limits (and usage tier): https://platform.openai.com/settings/organization/limits
- Monitor the OpenAI API usage (costs, credits and bills): https://platform.openai.com/settings/organization/usage
- Manually monitor / manage the stored files on the OpenAI server: https://platform.openai.com/storage
- Manually monitor / manage the started batches on the OpenAI server: https://platform.openai.com/batches

### Tests

- Run the available pytests:
  ```bash
  pytest -v gpt_batch_api/utils_test.py
  ```
  Verify that all tests passed.

- Test the token counting class `TokenEstimator`:
  ```bash
  python -m gpt_batch_api.tokens_test
  ```
  Verify that all predicted token totals are equal or close to the actual required number of tokens. OpenAI changes things from time to time, so `TokenEstimator` may occasionally require updates in order to be accurate, especially for new models.

## Implementing a Custom Task

In order to define and run your own task, **refer to the example task implementations in [task_manager_demo.py](task_manager_demo.py)**, including the `main()` function and how the tasks are run.

The general steps to creating your own tasks are:

1) Read the documentation comments at the beginning of the `TaskManager` class, which outline which methods should be overridden, what sources of command line arguments are possible, and what simple properties the design of the task state, task output, and request metadata format need to satisfy.
2) Design/plan (e.g. on paper) the task-specific data format of the task state, task output, and request metadata format, so that all properties are satisfied.
3) If structured outputs are to be used in the requests, define a `pydantic.BaseModel` for the JSON schema that the LLM responses should strictly adhere to.
4) Decide on a task output file class (e.g. `DataclassOutputFile`, `DataclassListOutputFile`, or a custom `TaskOutputFile` implementation) and possibly define an appropriate subclass (e.g. to specify `Dataclass`), and define any associated dataclasses, pydantic models, and such.
5) Implement a custom task-specific subclass of `TaskManager`, given all the information from the previous steps. The subclass often needs to load data from file as input for generating the required requests and completing the task. This can be implemented inside the subclass, or can be implemented in separate code that e.g. then just passes pre-loaded data or a data loader class to the __init__ method of the `TaskManager` subclass. Refer to the documentation within each of the methods to override in the `TaskManager` class source code.
6) Ensure Python logging is configured appropriately (e.g. see `utils.configure_logging()`, or otherwise use `utils.ColorFormatter` manually to help ensure that warnings and errors stand out in terms of color).
7) Use `argparse` or `hydra` to configure command line arguments and pass them to the custom `TaskManager` subclass on init (refer to `main()` in `task_manager_demo.py`, and `config/gpt_batch_api.yaml`).
8) Run the custom task manager by constructing an instance of the class and calling `run()`.

### Running a Custom Task

When a custom task has been implemented and it is time to test it out, it is usually not such a good idea to just immediately attempt a full costly run and see what happens. Usually finetuning the implementation of a task to perfection, e.g. due to prompt engineering or coding aspects, is an iterative process. Below is an enumerated list of recommended safe steps to try out for each new implemented task, in order to ensure that everything is working perfectly correctly before committing more resources to larger or full runs. We assume:
- The custom task manager has, as intended, been wrapped into a script that uses `argparse` (including also `--model`) to configure the task manager and GPT requester, just like `task_manager_demo.py` does. If `hydra` (supported) or a direct programmatic interface is being used instead for the configuration parameters, then the commands below can easily be adjusted to the appropriate form.
- Your generic command to run the custom script is denoted `python ARGS` below, which in the case of `task_manager_demo.py`  would for example be `python -m gpt_batch_api.task_manager_demo --task char_codes`.
- Refer to `MODELARGS` in [Run Commands](#run-commands), and `OPENAI_API_KEY` and `WANDB_API_KEY` in [Task Manager Demo](#task-manager-demo) and [Useful Hints](#useful-hints).

Here are the recommended safe steps:
1) List all available command line arguments:
   ```bash
   python ARGS --help
   ```
2) Assuming the task does not exist yet (no task run files exist), initialize a new task but do not run it (task metadata is taken and fixed beyond here unless using `--reinit_meta`):
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --no_run [--max_completion_tokens NUM] [--completion_ratio RATIO] [--temperature TEMP] [--top_p MASS] [--opinions_min NUM] [--opinions_max NUM] [--confidence RATIO] ...
   ```
3) Generate a batch of 100 requests without actually executing the batch:
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --max_session_requests 0 --max_batch_requests 100 --max_unpushed_batches 1
   ```
4) Show 10 verbose examples of generated requests without actually executing them (requires that some batches have already been generated and saved to disk):
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --max_session_requests 10 --max_batch_requests 100 --max_unpushed_batches 1 --force_direct --direct_verbose always --dryrun
   ```
5) Show 10 verbose examples of generated requests and responses using the direct API:
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --max_session_requests 10 --max_batch_requests 100 --max_unpushed_batches 1 --force_direct --direct_verbose always
   ```
6) Execute 100 requests using the direct API, and show those examples verbosely that had a warning or error:
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --max_session_requests 100 --max_batch_requests 100 --max_unpushed_batches 1 --force_direct --direct_verbose warn
   ```
7) Execute 250 requests using the batch API:
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --max_session_requests 400 --max_batch_requests 250 --max_unpushed_batches 1
   ```
   At this point we can decide on suitable initial values for `--max_completion_tokens` and `--completion_ratio`. To do this, search the output of the command above for `max # completion tokens` (overall metrics) and `max # tokens` (per batch), where `#` are some numbers, and decide on a suitable value of `--max_completion_tokens` that has a significant safety margin to the values seen. Also search the command outputs for `completion ratio` (overall metrics and per batch), and decide on a suitable value for `--completion_ratio`, factoring in whatever change you just made to `--max_completion_tokens` of course, as the completion ratio is a ratio of that. We can then update the (otherwise usually fixed) metadata of the task run using:
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --no_run --reinit_meta --max_completion_tokens NUM --completion_ratio RATIO [--temperature TEMP] [--top_p MASS] [--opinions_min NUM] [--opinions_max NUM] [--confidence RATIO] ...  # <-- CAUTION: Other than updating --max_completion_tokens and --completion_ratio, make sure to be consistent with point 2 as --reinit_meta always updates ALL task meta variables
   ```
8) Execute 1 USD worth of fresh (thus wipe ongoing) requests using the batch API (or 1 of whatever currency or unit `MODELARGS` used for `--cost_*`):
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --only_process  # <-- If there are any unfinished batches busy on the remote let them finish and be processed (we do not want to unnecessarily wipe them if they are already costing money)
   python ARGS "${MODELARGS[@]}" --no_wandb --wipe_requests --max_session_cost 1.00 --max_batch_cost 1.00 --max_unpushed_batches 1
   ```
   At this point you can refine the estimated value of the completion ratio, if required. Search the command outputs for the `completion ratio` of any newly executed batches, and reinitialize the task metadata as in point 7.
9) Reset the entire task to scratch (throwing away any results obtained so far, but ensuring a clean slate for a future full run):
   ```bash
   python ARGS "${MODELARGS[@]}" --no_wandb --wipe_task --no_run
   ```

We can now run the custom task normally to completion, making use of the wandb integration for logging and monitoring. Assuming we wish to respect a batch queue limit of 60,000,000 tokens (= 60000 ktokens) at a time (refer to the [OpenAI tier rate limits](https://platform.openai.com/settings/organization/limits)), we can do:
```bash
python ARGS "${MODELARGS[@]}" --wandb_name MY_BATCH_TASK --max_remote_ktokens 60000  # <-- CAUTION: Change MY_BATCH_TASK to a reasonable task name a sit should appear in wandb
```
The above command can be interrupted with `Ctrl+C` and restarted at any time, and it will safely and robustly just pick up wherever it left off without losing anything (assuming the custom task has been correctly and revertibly implemented like described in the documentation, and as demonstrated in `task_manager_demo.py`). Any already uploaded and started remote batches will continue to process even when the script is not running.

If some samples fail permanently (the maximum number of internal retries was reached, see `--max_retries`), they can be reset ('wiped') and retried a further `--max_retries` number of times using:
```bash
python ARGS "${MODELARGS[@]}" --wandb_name MY_BATCH_TASK --wandb_run_id RUN_ID --max_remote_ktokens 60000 --wipe_failed --max_retries 4  # <-- CAUTION: Appropriately set MY_BATCH_TASK and RUN_ID
```
Note that a further 4 allowed retries means that each failed sample gets a further 5 attempts. Note also that the command above additionally demonstrates how a wandb run can be resumed across invocations of the Python script, by setting the wandb run ID argument to that of the original wandb run that resulted from the first script invocation.

At any time if a task script instance is not running, you can check the task's overall status using something like:
```bash
python ARGS "${MODELARGS[@]}" --max_remote_ktokens 60000 --no_run
```

The following table shows a variety of commonly useful script arguments for debugging, applying code changes, recovering from errors/batch failures, and such:

| **Argument(s)**                                                    | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|--------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--task_dir /PATH/TO/DIR`                                          | Change the directory in which files are created to store the ongoing and final state, and output data of the command (recommended if installed via pip)                                                                                                                                                                                                                                                                                                                                                                                              |
| `--no_wandb`                                                       | Disable wandb logging                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `--wandb_project NAME`                                             | Set the desired target wandb project for logging                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `--wandb_name NAME`                                                | Set a custom fixed name for the target wandb logging run                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `--wandb_run_id RUN_ID`                                            | Resume/append to a wandb run (specified by the desired run ID), or create it if it does not exist (auto-generated run IDs are typically lowercase alphanumeric strings of length 8)                                                                                                                                                                                                                                                                                                                                                                  |
| `--dryrun`                                                         | Prevent any API calls or changes to saved disk state, and just show what would be done based on the current task state                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `--force_direct --direct_verbose always --max_session_requests 20` | Force only 20 requests to be made with the direct API, and print the requests and responses verbosely for debugging purposes (when combined with `--dryrun` only prints the requests that *would* have been made)                                                                                                                                                                                                                                                                                                                                    |
| `--max_remote_ktokens 90 --max_batch_ktokens 90`                   | Configure how many kilo-tokens can be pending in the batch queue at any one time in order to respect the OpenAI usage tier limitations (Note: `--max_batch_ktokens` must always be less than or equal to `--max_remote_ktokens`, otherwise no batch would fit on the remote)                                                                                                                                                                                                                                                                         |
| `--max_batch_requests 50`                                          | Limit the batch size in terms of number of requests per batch (also, num batches/num requests/num tokens/cost/MB size can be limited as appropriate for each batch, the remote server, each session, and/or the entire task)                                                                                                                                                                                                                                                                                                                         |
| `--max_remote_batches 0 --max_unpushed_batches 3`                  | Generate up to 3 local batches without letting any of them be pushed (e.g. allowing them to be manually inspected on disk without pushing)                                                                                                                                                                                                                                                                                                                                                                                                           |
| `--max_retries 5`                                                  | Adjust how often a request is automatically retried (if a retryable error occurs) before it is declared as failed                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `--min_pass_ratio 0.8`                                             | At minimum what ratio of the requests in a batch need to be successful in order for the batch to be declared as 'passed' (too many consecutive non-passed batches for safety lead to an error and aborting the run, see `--max_pass_failures`)                                                                                                                                                                                                                                                                                                       |
| `--process_failed_batches 2 [--retry_fatal_requests]`              | If there are failed batches then the task aborts for safety (as manual intervention/code changes/sanity checking is probably required). This parameter allows up to a certain number of failed batches to be force-processed anyway, thereby allowing the task to proceed and potentially recover. If `--retry_fatal_requests` is also supplied, then requests that received fatal errors will be allowed to be retried (normally they are not, as fatal errors are ones where it is not expected that a retry has a chance of resolving the issue). |
| `--only_process`                                                   | Wait for and allow all pushed batches to complete and be processed without generating, commiting or pushing any new requests (i.e. no new work is generated or scheduled)                                                                                                                                                                                                                                                                                                                                                                            |
| `--reinit_meta`                                                    | Force the task metadata to be updated (usually task metadata is only captured once when the task is created/initialized), for example to change the request model, temperature, hyperparameters for parsing, or such (you must ensure that your task implementation can deal with whatever of these parameters you are changing)                                                                                                                                                                                                                     |
| `--wipe_requests`                                                  | Wipe all ongoing requests (e.g. useful if you changed the request generation and want to reconstruct all the requests in the queue, local batches, and remote). It is recommended to execute a separate run first with `--only_process` to avoid unnecessarily losing already-pushed requests and responses.                                                                                                                                                                                                                                         |
| `--wipe_failed`                                                    | Wipe all ongoing requests and failed samples, allowing them to be attempted again (with the full number of retries available again). It is recommended to execute a separate run first with `--only_process` to avoid unnecessarily losing already-pushed requests and responses.                                                                                                                                                                                                                                                                    |
| `--wipe_task`                                                      | Wipe entire task and start completely from scratch (can be combined with `--no_run` in order to not actually run the task after wiping)                                                                                                                                                                                                                                                                                                                                                                                                              |

## Wandb Logging

Wandb is used by default by the `TaskManager` and `GPTRequester` classes for logging and monitoring, unless explicitly disabled using `wandb=False` (`--no_wandb`). You can control the wandb project name using `wand_project='NAME'` (`--wandb_project NAME`, otherwise the default is `gpt_batch_api`), and if need be control the wandb entity using `wandb_entity='NAME'` (`--wandb_entity NAME`). If you wish to resume and continue a previous wandb run (essentially append to it), then find out its run ID (e.g. `a1dr4wwa`) and use something like `wandb_run_id='a1dr4wwa'` (`--wandb_run_id a1dr4wwa`).

In order to get the most out of the wandb logging, it is recommended to use the preconfigured `gpt_batch_api` saved view for the project being logged to (to show and organize the most useful stuff to see/monitor). Refer to [this example](https://wandb.ai/pallgeuer/gpt_batch_api_demo?nw=u38mn3ltgfn) to see how the saved view looks like. Here are the steps for configuring its use for your own wandb project that visualizes output generated by this library:
- Ensure your target wandb project exists and has at least one run in it. Easiest is just to start your first desired run.
- Copy the preconfigured `gpt_batch_api` saved view to the project workspace (assuming the project is `ENTITY/PROJECT`):
  ```bash
  python -m gpt_batch_api.wandb_configure_view --dst_entity ENTITY --dst_project PROJECT
  ```
  In most cases the default `PROJECT` name of `gpt_batch_api` is being used. If this is the case, you can just do:
  ```bash
  python -m gpt_batch_api.wandb_configure_view --dst_entity ENTITY
  ```
  Refer to the help for a more general understanding of what the script can do:
  ```bash
  python -m gpt_batch_api.wandb_configure_view --help
  ```
- Open your target wandb project in the web browser, i.e. `https://wandb.ai/ENTITY/PROJECT`
- Click on the icon with the three horizontal lines at the top-left of the workspace and switch to the saved view called *GPT Batch API* (should exist if the `wandb_configure_view script` worked correctly).
- Click the *Copy to my workspace* button at the top-right of the workspace. This updates your personal workspace in the project to look like the desired preconfigured view

You can search the code for the name of logged variables (possibly including the prefix like `Batch/` for example) to get an idea of what the variable represents (e.g. search for `num_final_none` in the code).

## Local Inference

For purposes of reproducibility, it may be of interest to run a task with a locally hosted LLM instead of one of OpenAI's models. While locally hosted LLMs generally do not support OpenAI's [Batch API](https://platform.openai.com/docs/guides/batch), they often support OpenAI's [direct API](https://platform.openai.com/docs/api-reference/chat). As such, when forced to exclusively use the direct API (`--force_direct`), this library is completely compatible with locally hosted LLMs!

Suppose we would normally run a task using:
```bash
python -m gpt_batch_api.task_manager_demo --task_dir /tmp/gpt_batch_api_tasks --task utterance_emotion --model gpt-4o-mini-2024-07-18 --cost_input_direct_mtoken 0.150 --cost_input_cached_mtoken 0.075 --cost_input_batch_mtoken 0.075 --cost_output_direct_mtoken 0.600 --cost_output_batch_mtoken 0.300
```
Then if we are using, for example, [Ollama](https://ollama.com) to host an LLM at `http://IP:PORT`, we can run the task locally and directly using:
```bash
python -m gpt_batch_api.task_manager_demo --task_dir /tmp/gpt_batch_api_tasks --task utterance_emotion --client_base_url http://IP:PORT/v1 --model llama3.1:70b --force_direct --cost_input_direct_mtoken 0 --cost_input_cached_mtoken 0 --cost_input_batch_mtoken 0 --cost_output_direct_mtoken 0 --cost_output_batch_mtoken 0  # <-- CAUTION: Set IP and PORT to appropriate values
```
Instead of specifying `--client_base_url`, you can alternatively just set the `OPENAI_BASE_URL` environment variable (see [Useful hints](#useful-hints)):
```bash
export OPENAI_BASE_URL="http://IP:PORT/v1"  # <-- CAUTION: Set IP and PORT to appropriate values
```
Local LLMs hosted by means other than Ollama should work similarly out of the box, provided the appropriate direct API is exposed.

## Citation
This repository is licensed under the [GNU GPL v3 license](LICENSE.md). Please give appropriate attribution to Philipp Allgeuer and this repository if you use it for your own purposes, publications, theses, reports, or derivative works. Thanks!
