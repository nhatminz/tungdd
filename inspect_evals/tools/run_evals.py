from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

import backoff
from inspect_ai.model import Model

from inspect_evals.metadata import (
    ExternalEvalMetadata,
    InternalEvalMetadata,
    TaskMetadata,
    load_listing,
)

if sys.version_info < (3, 11):
    if not TYPE_CHECKING:
        sys.exit("Requires Python 3.11 or higher (ExceptionGroup is used).")

    class ExceptionGroup(Exception):
        def __init__(self, message: str, exceptions: list[Exception]) -> None: ...


logger = logging.getLogger(__name__)

MISSING_MODULE_REASON = "ModuleNotFoundError"
EVAL_TIMEOUT_MINS = 5
DEFAULT_SAMPLE_LIMIT = 0
DEFAULT_MODEL = None
PLACEHOLDER_VALUE = "placeholder"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
OFFLINE_SKIP_REASON = "Eval metadata declares runtime Internet access is required"
OFFLINE_PROXY_URL = "http://127.0.0.1:9"
LOCAL_MODEL_ROLES = (
    "chat",
    "conartist",
    "evaluator",
    "expander",
    "grader",
    "judge",
    "judger",
    "manipulatee",
    "manipulator",
    "mark",
    "persuadee",
    "persuader",
    "rater",
    "refusal_judge",
    "user",
    "victim",
)
OFFLINE_ERROR_PATTERNS = (
    "offline mode blocked outbound network access",
    "offlinemodeisenabled",
    "localentrynotfounderror",
    "network connectivity is disabled",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided",
    "couldn't reach",
    "cannot reach",
    "failed to establish a new connection",
    "failed to connect to 127.0.0.1 port 9",
    "clientconnectorerror",
    "requests.exceptions.connectionerror",
    "httpx.connecterror",
    "api connection error",
)
# Transient sqlite errors occur when Inspect's internal log buffer database
# fails to initialise under disk pressure or parallel execution. These are
# infrastructure errors (not eval bugs), so we retry rather than fail.
SQLITE_TRANSIENT_ERROR = "sqlite3.OperationalError: unable to open database file"
MAX_SMOKE_TEST_RETRIES = 3

TASK_SPECIFIC_ENV_VARS = {
    "cybench": {"CYBENCH_ACKNOWLEDGE_RISKS": "1"},
    "sandboxbench": {"SANDBOXBENCH_ACKNOWLEDGE_RISKS": "1"},
}

# TODO: These can be fixed
KNOWN_FAILURES = {
    "paperbench_score": "Nonstandard eval",
}

WINDOWS_HEAVY_EVAL_REASON = (
    "Too resource-heavy for the Windows smoke runner; see entry comment."
)

WINDOWS_LINUX_DOCKER_REASON = (
    "Requires a Linux Docker image; Windows GitHub runners use Windows-container "
    "mode and cannot launch Linux images."
)

KNOWN_WINDOWS_ONLY_FAILURES = {
    "swe_bench_verified_mini": MISSING_MODULE_REASON,
    "swe_bench": MISSING_MODULE_REASON,
    # The Windows smoke runner has run out of disk on every scheduled run
    # since 2026-05-01, exiting with STATUS_ACCESS_VIOLATION (3221225477).
    # Linux hits the 15-min eval-timeout before it would run out of disk.
    # See https://github.com/UKGovernmentBEIS/inspect_evals/pull/1615.
    "cybergym": WINDOWS_HEAVY_EVAL_REASON,
    # cti_realm spins up a Linux Docker sandbox (python:3.11-slim). Windows
    # GitHub runners run Docker in Windows-container mode and cannot launch
    # Linux images, producing: image operating system "linux" cannot be used
    # on this platform.
    "cti_realm_25": WINDOWS_LINUX_DOCKER_REASON,
    "cti_realm_50": WINDOWS_LINUX_DOCKER_REASON,
    "cti_realm_25_minimal": WINDOWS_LINUX_DOCKER_REASON,
    "cti_realm_25_seeded": WINDOWS_LINUX_DOCKER_REASON,
}

CI_ONLY_IGNORES: dict[str, str] = {}

MISSING_CREDENTIALS_REASON = "Missing required credentials"


@dataclass(frozen=True)
class TaskRunResult:
    """Outcome for one task invocation."""

    task_name: str
    descriptor: str
    status: str
    reason: str | None = None


def _has_kaggle_credentials() -> bool:
    """Returns True if non-placeholder kaggle keys are set. Used to skip eval smoke tests that require valid kaggle keys."""
    key = os.environ.get("KAGGLE_KEY")
    username = os.environ.get("KAGGLE_USERNAME")
    if key and username:
        if PLACEHOLDER_VALUE not in (key, username):
            return True
    return Path(os.path.expanduser("~/.kaggle/kaggle.json")).exists()


MISSING_CREDENTIALS_SKIPS: dict[str, str] = {
    **(
        {
            "mle_bench": MISSING_CREDENTIALS_REASON,
            "mle_bench_lite": MISSING_CREDENTIALS_REASON,
            "mle_bench_full": MISSING_CREDENTIALS_REASON,
        }
        if not _has_kaggle_credentials()
        else {}
    ),
}


# TODO: We want to make this as small as possible
# Fixing known failures, improving CI test running
# capability etc.
SKIPPABLE_TASKS = {
    **KNOWN_FAILURES,
    **(KNOWN_WINDOWS_ONLY_FAILURES if platform.system() == "Windows" else {}),
    **(CI_ONLY_IGNORES if os.environ.get("GITHUB_ACTIONS") else {}),
    **MISSING_CREDENTIALS_SKIPS,
}


def get_evals(
    predicate_regexp: str | None = None,
) -> list[ExternalEvalMetadata | InternalEvalMetadata]:
    evals = load_listing().evals

    # External evals live in upstream repositories and cannot be smoke-tested
    # locally — their task_path refers to files in the external repo, not this
    # one.  Filter them out so the runner only exercises internal evals.
    evals = [e for e in evals if not isinstance(e, ExternalEvalMetadata)]

    if predicate_regexp is None:
        return evals
    pattern = re.compile(predicate_regexp)
    filtered_evals: list[ExternalEvalMetadata | InternalEvalMetadata] = []
    for eval_meta in evals:
        if pattern.search(eval_meta.id) or pattern.search(eval_meta.path):
            filtered_evals.append(eval_meta)
            continue
        matching_tasks = [task for task in eval_meta.tasks if pattern.search(task.name)]
        if matching_tasks:
            filtered_evals.append(
                eval_meta.model_copy(update={"tasks": matching_tasks})
            )
    return filtered_evals


def offline_error_reason(output: str) -> str | None:
    """Return a clear skip reason for an expected offline resource failure."""
    normalized = output.lower()
    matched = next(
        (pattern for pattern in OFFLINE_ERROR_PATTERNS if pattern in normalized),
        None,
    )
    if matched is None:
        return None
    return f"Offline dependency/resource unavailable ({matched})"


def offline_environment(
    vllm_base_url: str | None = None, vllm_api_key: str | None = None
) -> dict[str, str]:
    """Environment overrides that prevent remote access but permit local vLLM."""
    api_key = vllm_api_key or "inspectai"
    environment = {
        "INSPECT_EVALS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "UV_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "DO_NOT_TRACK": "1",
        "HTTP_PROXY": OFFLINE_PROXY_URL,
        "HTTPS_PROXY": OFFLINE_PROXY_URL,
        "ALL_PROXY": OFFLINE_PROXY_URL,
        "http_proxy": OFFLINE_PROXY_URL,
        "https_proxy": OFFLINE_PROXY_URL,
        "all_proxy": OFFLINE_PROXY_URL,
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
        "VLLM_API_KEY": api_key,
        # Override real provider credentials so an undeclared external model
        # cannot use a credential inherited from the caller's shell.
        "ANTHROPIC_API_KEY": PLACEHOLDER_VALUE,
        "GOOGLE_API_KEY": PLACEHOLDER_VALUE,
        "OPENAI_API_KEY": api_key,
    }
    if vllm_base_url:
        environment["VLLM_BASE_URL"] = vllm_base_url
        # Unconfigurable hard-coded OpenAI models still stay on the local host.
        environment["OPENAI_BASE_URL"] = vllm_base_url
    return environment


def validate_vllm_base_url(base_url: str) -> str:
    """Validate and normalize a loopback OpenAI-compatible endpoint URL."""
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--vllm-base-url must use http:// or https://")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--vllm-base-url must target a loopback host in offline mode")
    if not parsed.port:
        raise ValueError("--vllm-base-url must include an explicit port")
    return normalized


def check_vllm_endpoint(base_url: str, model_name: str, api_key: str | None) -> None:
    """Confirm the local server is reachable and exposes the requested model."""
    request = Request(f"{base_url}/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Local vLLM endpoint is unavailable at {base_url}: {exc}"
        ) from exc

    available_models = {
        item.get("id") for item in payload.get("data", []) if isinstance(item, dict)
    }
    if model_name not in available_models:
        available = ", ".join(sorted(str(model) for model in available_models))
        raise RuntimeError(
            f"Model {model_name!r} is not served by {base_url}; available: {available}"
        )


def handle_accepted_errors(
    error: subprocess.CalledProcessError,
    task_name: str,
    descriptor: str,
    offline: bool = False,
) -> str:
    logger.info(
        f"Found error on {descriptor}: {error=}. Checking if error is accepted..."
    )

    if any(skippable_task == task_name for skippable_task in SKIPPABLE_TASKS):
        reason = SKIPPABLE_TASKS[task_name]
        logger.warning(f"SKIPPED {descriptor}: {reason}. Would get {error=}...")
        return reason

    if offline and (reason := offline_error_reason(error.output or "")):
        logger.warning(f"SKIPPED {descriptor}: {reason}")
        return reason

    gated_dataset_error = (
        "DatasetNotFoundError" in error.output
        and "is a gated dataset on the Hub" in error.output
    ) or "huggingface_hub.errors.GatedRepoError" in error.output

    optional_dependency_missing_error = "require_optional_dependency" in error.output

    def ignored_error(reason: str) -> str:
        return f"Error on {descriptor} ({error=}) found to be {reason}... ignoring"

    accepted_errors = {
        gated_dataset_error: ignored_error("gated dataset error"),
        optional_dependency_missing_error: ignored_error("optional dependency error"),
    }

    for accepted_error, reason in accepted_errors.items():
        if accepted_error:
            logger.warning(reason)
            return reason

    logger.warning(
        f"Error on {descriptor} ({error=}) is not considered acceptable... "
        f"{error.output=}"
    )
    raise error


def _is_transient_subprocess_error(e: subprocess.CalledProcessError) -> bool:
    """Check if a subprocess error is transient and should be retried."""
    return SQLITE_TRANSIENT_ERROR in (e.output or "")


def _is_retriable_error(e: Exception) -> bool:
    """Decide whether backoff should give up (True = give up, False = retry)."""
    if isinstance(e, subprocess.CalledProcessError):
        return not _is_transient_subprocess_error(e)
    return True


@backoff.on_exception(
    backoff.expo,
    exception=subprocess.CalledProcessError,
    giveup=_is_retriable_error,
    max_tries=MAX_SMOKE_TEST_RETRIES,
    jitter=backoff.full_jitter,
)
def _run_eval(
    run_eval_command: list[str],
    task_name: str,
    eval_timeout_mins: int,
    offline: bool = False,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
) -> None:
    environment = {
        **os.environ,
        **TASK_SPECIFIC_ENV_VARS.get(task_name, {}),
        **inject_placeholder_llm_keys(),
        **inject_gdm_placeholder_env_vars(),
    }
    if offline:
        environment.update(offline_environment(vllm_base_url, vllm_api_key))
    elif vllm_base_url:
        environment["VLLM_BASE_URL"] = vllm_base_url
        environment["VLLM_API_KEY"] = vllm_api_key or "inspectai"

    subprocess.run(
        run_eval_command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        timeout=eval_timeout_mins * 60,
    )


def model_task_args(
    eval_meta: ExternalEvalMetadata | InternalEvalMetadata,
    task_name: str,
    model: str,
) -> dict[str, str]:
    """Map every ``Model``-typed ``@task`` param to one model reference.

    At ``--limit 0`` no sample runs, so models are never *called* — they only
    need to *construct*. Judge/grader models instantiated eagerly at
    construction time (e.g. writingbench) are the main reason a task otherwise
    needs real provider credentials or a minimum provider package version.
    Overriding them with mockllm keeps the smoke test measuring the eval's own
    code rather than provider availability.

    Params are matched by *type*, so they are caught whatever they are named.
    A list-typed param uses YAML flow-list syntax so Inspect's ``-T`` parser
    yields a list. Only internal evals are handled; an import/inspection
    failure yields ``{}`` so the eval runs unchanged.
    """
    if not isinstance(eval_meta, InternalEvalMetadata):
        return {}
    try:
        module = importlib.import_module(f"inspect_evals.{eval_meta.id}")
        parameters = inspect.signature(getattr(module, task_name)).parameters
    except Exception as e:
        logger.warning(
            f"Could not inspect {eval_meta.id}/{task_name} for Model-typed "
            f"params; running it without model-parameter overrides ({e!r})"
        )
        return {}

    def contains_model(hint: Any) -> bool:
        return hint is Model or any(contains_model(arg) for arg in get_args(hint))

    task_args: dict[str, str] = {}
    for name, param in parameters.items():
        hint = param.annotation
        if hint is inspect.Parameter.empty or not contains_model(hint):
            continue
        is_list = get_origin(hint) is list or any(
            get_origin(arg) is list for arg in get_args(hint)
        )
        task_args[name] = f"[{model}]" if is_list else model
    return task_args


def mock_model_task_args(
    eval_meta: ExternalEvalMetadata | InternalEvalMetadata, task_name: str
) -> dict[str, str]:
    """Map every ``Model``-typed task parameter to ``mockllm/model``."""
    return model_task_args(eval_meta, task_name, "mockllm/model")


def task_accepts_parameter(
    eval_meta: ExternalEvalMetadata | InternalEvalMetadata,
    task_name: str,
    parameter: str,
) -> bool:
    """Return whether an internal task explicitly accepts a parameter."""
    if not isinstance(eval_meta, InternalEvalMetadata):
        return False
    try:
        module = importlib.import_module(f"inspect_evals.{eval_meta.id}")
        return parameter in inspect.signature(getattr(module, task_name)).parameters
    except Exception as e:
        logger.warning(
            f"Could not inspect {eval_meta.id}/{task_name} for {parameter!r}; "
            f"treating the parameter as unsupported ({e!r})"
        )
        return False


def smoke_test(
    task_name: str,
    task_ref: str,
    descriptor: str,
    eval_timeout_mins: int,
    subprocess_semaphore: threading.BoundedSemaphore,
    limit: int,
    model: str | None,
    package_path: str | None = None,
    task_model_args: dict[str, str] | None = None,
    model_base_url: str | None = None,
    model_roles: tuple[str, ...] = (),
    offline: bool = False,
    vllm_api_key: str | None = None,
) -> TaskRunResult:
    """
    Invoke eval task.

    An eval task will either:

    (1) Succeed cleanly
    (2) Fail

    Failures are partitioned by:

    (1) Timeouts
    (2) Acceptable errors (see: `handle_accepted_errors`)
    (3) Transient errors that are retried (e.g. sqlite infrastructure errors)
    (4) Unexpected errors (everything else)

    Acceptable errors are partitioned by:

    (1) Eval-specific errors marked to fail a-priori (see: `SKIPPABLE_TASKS`)
    (2) Errors related to gated repos and dependencies
    """
    run_eval_command = ["uv", "run"]
    if package_path:
        run_eval_command.extend(["--directory", package_path])
    if offline:
        run_eval_command.extend(["--offline", "--no-sync"])
    run_eval_command.extend(["inspect", "eval", task_ref, "--limit", str(limit)])
    for name, value in (task_model_args or {}).items():
        run_eval_command.extend(["-T", f"{name}={value}"])
    if model:
        run_eval_command.extend(["--model", model])
    if model_base_url:
        run_eval_command.extend(["--model-base-url", model_base_url])
    for role in model_roles:
        role_spec = json.dumps(
            {"model": model, "model_args": {"base_url": model_base_url}}
        )
        run_eval_command.extend(["--model-role", f"{role}={role_spec}"])
    if offline:
        run_eval_command.extend(["--max-retries", "1"])
    if model_base_url:
        run_eval_command.append("--log-model-api")

    logger.info(f"Testing {descriptor}")
    logger.debug("Command: %s", subprocess.list2cmdline(run_eval_command))

    try:
        with subprocess_semaphore:
            _run_eval(
                run_eval_command,
                task_name,
                eval_timeout_mins,
                offline=offline,
                vllm_base_url=model_base_url,
                vllm_api_key=vllm_api_key,
            )

    except subprocess.CalledProcessError as e:
        reason = handle_accepted_errors(e, task_name, descriptor, offline=offline)
        return TaskRunResult(task_name, descriptor, "skipped", reason)

    except subprocess.TimeoutExpired as e:
        reason = f"Timed out after {eval_timeout_mins} minutes"
        logger.warning(f"SKIPPED {descriptor}: {reason}. {e.output=}")
        return TaskRunResult(task_name, descriptor, "skipped", reason)

    if task_name in SKIPPABLE_TASKS:
        logger.warning(
            f"Task {task_name=} is in SKIPPABLE_TASKS, but should be removed due to running successfully."
        )

    logger.info(f"Succeeded: {descriptor}")
    return TaskRunResult(task_name, descriptor, "succeeded")


def run_eval_job(
    eval_meta: ExternalEvalMetadata | InternalEvalMetadata,
    eval_index: int,
    eval_timeout_mins: int,
    within_eval_concurrency: bool,
    subprocess_semaphore: threading.BoundedSemaphore,
    limit: int,
    model: str | None,
    mock_models: bool,
    model_base_url: str | None = None,
    model_roles: tuple[str, ...] = (),
    offline: bool = False,
    vllm_api_key: str | None = None,
    offline_only: bool = False,
) -> list[TaskRunResult]:
    """Run one eval's tasks, serially or in parallel per ``within_eval_concurrency``."""
    total = len(eval_meta.tasks)
    internet_required = (
        offline
        and eval_meta.runtime_metadata is not None
        and eval_meta.runtime_metadata.requires_internet is True
    )
    results: list[TaskRunResult] = []
    selected_tasks: list[tuple[int, TaskMetadata, bool]] = []
    for task_index, task in enumerate(eval_meta.tasks):
        descriptor = (
            f"eval {eval_index} ({eval_meta.id}) "
            f"task {task_index + 1} of {total} ({task.name})"
        )
        supports_offline_only = offline_only and task_accepts_parameter(
            eval_meta, task.name, "offline_only"
        )
        if internet_required and not supports_offline_only:
            logger.warning(f"SKIPPED {descriptor}: {OFFLINE_SKIP_REASON}")
            results.append(
                TaskRunResult(
                    task_name=task.name,
                    descriptor=descriptor,
                    status="skipped",
                    reason=OFFLINE_SKIP_REASON,
                )
            )
            continue
        if supports_offline_only:
            logger.info("Applying offline_only sample filter to %s", descriptor)
        selected_tasks.append((task_index, task, supports_offline_only))

    if not selected_tasks:
        return results

    package_path = (
        eval_meta.package_path if isinstance(eval_meta, InternalEvalMetadata) else None
    )

    def _task_ref(task: TaskMetadata) -> str:
        # Registry entries run via file-path (upstream's task file); internal
        # evals use the entry-point prefix exposed by the installed package.
        if isinstance(eval_meta, ExternalEvalMetadata):
            return f"{task.task_path}@{task.name}"
        return f"inspect_evals/{task.name}"

    invocations = []
    for task_index, task, supports_offline_only in selected_tasks:
        task_args = (
            model_task_args(eval_meta, task.name, model)
            if model and model_base_url
            else mock_model_task_args(eval_meta, task.name)
            if mock_models
            else {}
        )
        if supports_offline_only:
            task_args["offline_only"] = "true"
        invocations.append(
            (
                task.name,
                _task_ref(task),
                (
                    f"eval {eval_index} ({eval_meta.id}) "
                    f"task {task_index + 1} of {total} ({task.name})"
                ),
                eval_timeout_mins,
                subprocess_semaphore,
                limit,
                model,
                package_path,
                task_args,
                model_base_url,
                model_roles,
                offline,
                vllm_api_key,
            )
        )
    workers = len(invocations) if within_eval_concurrency else 1
    with ThreadPoolExecutor(max_workers=workers) as exec:
        futures = [exec.submit(smoke_test, *args) for args in invocations]
        for f in as_completed(futures):
            results.append(f.result())
    return results


def run_evals(
    evals: list[ExternalEvalMetadata | InternalEvalMetadata],
    exec: ThreadPoolExecutor,
    eval_start_index: int,
    eval_timeout_mins: int,
    fail_fast: bool,
    within_eval_concurrency: bool,
    subprocess_semaphore: threading.BoundedSemaphore,
    limit: int,
    model: str | None,
    mock_models: bool,
    model_base_url: str | None = None,
    model_roles: tuple[str, ...] = (),
    offline: bool = False,
    vllm_api_key: str | None = None,
    offline_only: bool = False,
) -> list[TaskRunResult]:
    tasks = [
        exec.submit(
            run_eval_job,
            e,
            eval_start_index + i,
            eval_timeout_mins,
            within_eval_concurrency,
            subprocess_semaphore,
            limit,
            model,
            mock_models,
            model_base_url,
            model_roles,
            offline,
            vllm_api_key,
            offline_only,
        )
        for i, e in enumerate(evals)
    ]
    results: list[TaskRunResult] = []
    try:
        for task in as_completed(tasks):
            results.extend(task.result())
    except subprocess.CalledProcessError as final_error:
        logger.warning("Found unexpected error (stdout + stderr)")
        for line in final_error.output.splitlines():
            logger.info(line)
        if fail_fast:
            logger.warning("Cancelling the remaining scheduled tasks")
            for task in tasks:
                task.cancel()
        raise
    return results


def inject_placeholder_llm_keys() -> dict[str, str]:
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "SPOONACULAR_API_KEY",
        "AICROWD_API_KEY",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
    ]
    # Treat present-but-empty vars as unset and inject placeholder.
    return {key: PLACEHOLDER_VALUE for key in keys if not os.environ.get(key)}


def inject_gdm_placeholder_env_vars() -> dict[str, str]:
    return {
        "EMPLOYEE_NAME": PLACEHOLDER_VALUE,
        "GDM_SELF_PROLIFERATION_DEV": PLACEHOLDER_VALUE,
        "PROJECT_ID": PLACEHOLDER_VALUE,
        "EMPLOYEE_WEBSITE": PLACEHOLDER_VALUE,
        "EMAIL_ADDRESS": PLACEHOLDER_VALUE,
    }


T = TypeVar("T")


def chunks(lst: list[T], n: int) -> list[list[T]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def clean() -> None:
    logger.info("Cleaning disk")
    # Inherit parent stdout/stderr so `make clean`'s own output (including
    # pre/post disk usage from tools/clean.py) surfaces to the smoke-test log.
    subprocess.run(
        "make clean DRY_RUN=false CLEAN_ARGS=--force",
        shell=True,
        check=True,
        text=True,
    )
    logger.info("Cache cleaning complete")


def validate_args(args: argparse.Namespace) -> None:
    if args.vllm_model and args.model:
        raise ValueError("--vllm-model cannot be combined with --model")

    if args.vllm_model:
        args.vllm_base_url = validate_vllm_base_url(
            args.vllm_base_url or DEFAULT_VLLM_BASE_URL
        )
        args.model = f"vllm/{args.vllm_model}"
    elif args.vllm_base_url or args.vllm_api_key:
        raise ValueError("--vllm-model is required with vLLM endpoint options")

    if args.offline is None:
        args.offline = bool(args.vllm_model)
    if args.offline_only is None:
        args.offline_only = bool(args.offline)

    if args.offline and args.limit > 0 and not args.vllm_model:
        raise ValueError(
            "Offline inference requires --vllm-model so all model calls can be "
            "routed to a local endpoint"
        )
    if args.limit > 0 and args.model is None:
        raise ValueError(
            "--model or --vllm-model must be provided when --limit > 0. "
            f"Found {args.model=}, {args.limit=}"
        )


def resolve_max_threads(n_threads: int | None) -> int:
    # Evals are I/O bound
    if n_threads is not None:
        return n_threads

    if (n_cpu := os.cpu_count()) is None:
        return 1

    return max(1, n_cpu // 2)


def smoke_test_all_evals() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Filter for a subset of inspect evals tasks.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        help="Raise on the first unexpected exception.",
        default=True,
    )
    parser.add_argument(
        "--eval-timeout-mins",
        type=int,
        help="The amount of time to have the eval run for until a timeout warning is logged.",
        default=EVAL_TIMEOUT_MINS,
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Show debug logs",
        default=False,
    )
    parser.add_argument(
        "--clean-after-chunk",
        action=argparse.BooleanOptionalAction,
        help="Purge caches with `make clean` after each chunk",
        default=False,
    )
    parser.add_argument(
        "--within-eval-concurrency",
        action=argparse.BooleanOptionalAction,
        help=(
            "Run tasks of the same eval in parallel. Default is serial: "
            "prevents races over shared state during Python imports "
            "(e.g. Kaggle config dir creation)."
        ),
        default=False,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Number of samples to run the eval for.",
        default=DEFAULT_SAMPLE_LIMIT,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to run the eval for.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=os.environ.get("INSPECT_EVALS_VLLM_BASE_URL"),
        help=(
            "Loopback OpenAI-compatible vLLM base URL. Defaults to "
            f"{DEFAULT_VLLM_BASE_URL} when --vllm-model is set, or reads "
            "INSPECT_EVALS_VLLM_BASE_URL."
        ),
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=os.environ.get("INSPECT_EVALS_VLLM_MODEL"),
        help=(
            "Model name exposed by the vLLM server; may also be set with "
            "INSPECT_EVALS_VLLM_MODEL."
        ),
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default=os.environ.get("INSPECT_EVALS_VLLM_API_KEY"),
        help=(
            "Optional vLLM API key; may also be set with INSPECT_EVALS_VLLM_API_KEY."
        ),
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Block public network access and skip online-only tasks/resources. "
            "Enabled by default when --vllm-model is used."
        ),
    )
    parser.add_argument(
        "--offline-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pass offline_only=true to tasks that support sample-level Internet "
            "filtering. Enabled by default in offline mode; unsupported online "
            "tasks are skipped in full."
        ),
    )
    parser.add_argument(
        "--mock-models",
        action=argparse.BooleanOptionalAction,
        help=(
            "Override every Model-typed task param (judge/grader models) with "
            "mockllm/model so tasks construct without real provider "
            "credentials. Use --no-mock-models to exercise real judges."
        ),
        default=True,
    )
    parser.add_argument("--n-threads", type=int, help="Number of threads", default=None)
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    validate_args(args)

    if not args.offline and not os.getenv("HF_TOKEN"):
        raise ValueError(
            "Requires a nonempty HF_TOKEN to get reasonable rate limiting policy."
        )

    if args.vllm_model and args.limit > 0:
        check_vllm_endpoint(args.vllm_base_url, args.vllm_model, args.vllm_api_key)

    max_threads = resolve_max_threads(args.n_threads)
    evals = get_evals(args.pred)
    n_evals = len(evals)
    n_tasks = sum(len(e.tasks) for e in evals)
    eval_chunks = chunks(evals, max_threads)
    logger.info(
        f"Testing {n_evals} evals ({n_tasks} tasks) in {len(eval_chunks)} chunk(s) "
        f"with {max_threads=}"
    )

    errors = []
    results: list[TaskRunResult] = []
    for i, chunk in enumerate(eval_chunks):
        logger.info(f"Running chunk {i + 1}/{len(eval_chunks)} ({len(chunk)} evals)")
        try:
            with ThreadPoolExecutor(max_workers=max_threads) as exec:
                results.extend(
                    run_evals(
                        chunk,
                        exec=exec,
                        eval_start_index=i * max_threads,
                        eval_timeout_mins=args.eval_timeout_mins,
                        fail_fast=args.fail_fast,
                        within_eval_concurrency=args.within_eval_concurrency,
                        subprocess_semaphore=threading.BoundedSemaphore(max_threads),
                        limit=args.limit,
                        model=args.model,
                        mock_models=args.mock_models,
                        model_base_url=args.vllm_base_url,
                        model_roles=LOCAL_MODEL_ROLES if args.vllm_model else (),
                        offline=args.offline,
                        vllm_api_key=args.vllm_api_key,
                        offline_only=args.offline_only,
                    )
                )
        except Exception as e:
            logger.exception(f"At least one eval in chunk {i + 1} failed")
            if args.fail_fast:
                raise
            errors.append(e)
        finally:
            if args.clean_after_chunk:
                clean()

    if errors:
        raise ExceptionGroup(f"{len(errors)} chunk(s) failed", errors)

    succeeded = sum(result.status == "succeeded" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    logger.info(
        f"Completed {n_evals} evals without unexpected errors: "
        f"{succeeded} task(s) succeeded, {skipped} task(s) skipped"
    )


if __name__ == "__main__":
    smoke_test_all_evals()
