"""Tests for tools/run_evals.py."""

# No `from __future__ import annotations` here: mock_model_task_args matches
# params by their resolved annotation objects, so PEP 563 string annotations
# would make fake_task's params invisible to it (as they would for a real eval).

import json
import logging
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
from inspect_ai.model import Model
from tools.run_evals import (
    OFFLINE_SKIP_REASON,
    PLACEHOLDER_VALUE,
    _has_kaggle_credentials,
    get_evals,
    handle_accepted_errors,
    inject_placeholder_llm_keys,
    mock_model_task_args,
    model_task_args,
    offline_environment,
    offline_error_reason,
    run_eval_job,
    smoke_test,
    validate_vllm_base_url,
)

from inspect_evals.metadata import (
    EvalRuntimeMetadata,
    ExternalEvalMetadata,
    Group,
    InternalEvalMetadata,
    TaskMetadata,
    load_eval_metadata,
)

FAKE_EVAL_ID = "fake_smoke_eval"


def fake_task(
    samples: int = 10,
    judge_model: str | Model = "openai/gpt-4o",
    grader_models: list[str | Model] | None = None,
    unannotated=None,
) -> None:
    pass


def offline_filter_task(offline_only: bool = False) -> None:
    pass


def internal_eval_meta() -> InternalEvalMetadata:
    return InternalEvalMetadata(
        title="Fake eval",
        description="Fake eval for tests.",
        id=FAKE_EVAL_ID,
        contributors=["tester"],
        tasks=[TaskMetadata(name="fake_task")],
        group=Group.CODING,
        version="1-A",
        external_assets=[],
    )


@pytest.fixture
def fake_eval_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType(f"inspect_evals.{FAKE_EVAL_ID}")
    module.fake_task = fake_task  # type: ignore[attr-defined]
    module.offline_filter_task = offline_filter_task  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, f"inspect_evals.{FAKE_EVAL_ID}", module)


def test_model_params_mocked_by_type(fake_eval_module: None) -> None:
    assert mock_model_task_args(internal_eval_meta(), "fake_task") == {
        "judge_model": "mockllm/model",
        "grader_models": "[mockllm/model]",
    }


def test_model_params_routed_to_local_model(fake_eval_module: None) -> None:
    assert model_task_args(internal_eval_meta(), "fake_task", "vllm/local-model") == {
        "judge_model": "vllm/local-model",
        "grader_models": "[vllm/local-model]",
    }


def test_writingbench_judge_model_detected() -> None:
    meta = load_eval_metadata("writingbench")
    assert mock_model_task_args(meta, "writingbench") == {
        "judge_model": "mockllm/model"
    }


def test_external_eval_returns_no_args() -> None:
    external = ExternalEvalMetadata(
        full_title="Fake external eval",
        common_title="Fake",
        description="Fake register entry for tests.",
        id="fake_external",
        contributors=["tester"],
        tasks=[TaskMetadata(name="fake_task", task_path="src/fake.py")],
        source={
            "repository_url": "https://github.com/example/fake",
            "repository_commit": "abc1234",
        },
    )
    assert mock_model_task_args(external, "fake_task") == {}


def test_uninspectable_eval_returns_no_args_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert mock_model_task_args(internal_eval_meta(), "fake_task") == {}
    assert any(FAKE_EVAL_ID in record.message for record in caplog.records)


def _capture_smoke_test_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr("tools.run_evals.smoke_test", lambda *args: calls.append(args))
    return calls


def _run_eval_job(mock_models: bool) -> tuple:
    return (
        internal_eval_meta(),
        0,
        1,
        False,
        threading.BoundedSemaphore(1),
        0,
        None,
        mock_models,
    )


def test_run_eval_job_passes_mock_model_args_when_enabled(
    fake_eval_module: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_smoke_test_calls(monkeypatch)
    run_eval_job(*_run_eval_job(mock_models=True))
    assert calls[0][8] == {
        "judge_model": "mockllm/model",
        "grader_models": "[mockllm/model]",
    }


def test_run_eval_job_skips_mock_model_args_when_disabled(
    fake_eval_module: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_smoke_test_calls(monkeypatch)
    run_eval_job(*_run_eval_job(mock_models=False))
    assert calls[0][8] == {}


def test_run_eval_job_routes_model_args_to_local_vllm(
    fake_eval_module: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_smoke_test_calls(monkeypatch)
    run_eval_job(
        internal_eval_meta(),
        0,
        1,
        False,
        threading.BoundedSemaphore(1),
        1,
        "vllm/local-model",
        True,
        model_base_url="http://127.0.0.1:8000/v1",
    )
    assert calls[0][8] == {
        "judge_model": "vllm/local-model",
        "grader_models": "[vllm/local-model]",
    }


def test_offline_internet_eval_is_skipped_before_subprocess(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    meta = internal_eval_meta().model_copy(
        update={"runtime_metadata": EvalRuntimeMetadata(requires_internet=True)}
    )
    calls = _capture_smoke_test_calls(monkeypatch)
    with caplog.at_level(logging.WARNING):
        results = run_eval_job(
            meta,
            0,
            1,
            False,
            threading.BoundedSemaphore(1),
            1,
            "vllm/local-model",
            False,
            offline=True,
            offline_only=True,
        )
    assert calls == []
    assert results[0].status == "skipped"
    assert results[0].reason == OFFLINE_SKIP_REASON


def test_offline_internet_eval_runs_sample_filter_aware_task(
    fake_eval_module: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = internal_eval_meta().model_copy(
        update={
            "tasks": [TaskMetadata(name="offline_filter_task")],
            "runtime_metadata": EvalRuntimeMetadata(requires_internet=True),
        }
    )
    calls = _capture_smoke_test_calls(monkeypatch)
    run_eval_job(
        meta,
        0,
        1,
        False,
        threading.BoundedSemaphore(1),
        1,
        "vllm/local-model",
        False,
        offline=True,
        offline_only=True,
    )
    assert calls[0][8] == {"offline_only": "true"}


def test_task_name_predicate_filters_to_matching_task() -> None:
    evals = get_evals("^tau2_telecom$")
    assert len(evals) == 1
    assert [task.name for task in evals[0].tasks] == ["tau2_telecom"]


def test_offline_error_is_classified_as_skip() -> None:
    reason = offline_error_reason(
        "OfflineNetworkError: offline mode blocked outbound network access"
    )
    assert reason is not None
    assert "unavailable" in reason


def test_offline_git_proxy_error_is_classified_as_skip() -> None:
    reason = offline_error_reason(
        "fatal: unable to access 'https://example.com/repo': "
        "Failed to connect to 127.0.0.1 port 9: Connection refused"
    )
    assert reason is not None
    assert "failed to connect to 127.0.0.1 port 9" in reason


def test_offline_environment_forces_local_routing() -> None:
    environment = offline_environment("http://127.0.0.1:8000/v1", "secret")
    assert environment["INSPECT_EVALS_OFFLINE"] == "1"
    assert environment["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert environment["VLLM_API_KEY"] == "secret"
    assert environment["ANTHROPIC_API_KEY"] == PLACEHOLDER_VALUE


def test_vllm_base_url_must_be_loopback() -> None:
    assert (
        validate_vllm_base_url("http://localhost:8000/v1/")
        == "http://localhost:8000/v1"
    )
    with pytest.raises(ValueError, match="loopback"):
        validate_vllm_base_url("https://api.openai.com/v1")


def test_smoke_test_routes_model_roles_to_local_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def capture(command: list[str], *_args, **_kwargs) -> None:
        commands.append(command)

    monkeypatch.setattr("tools.run_evals._run_eval", capture)
    result = smoke_test(
        "fake_task",
        "inspect_evals/fake_task",
        "fake descriptor",
        1,
        threading.BoundedSemaphore(1),
        1,
        "vllm/local-model",
        model_base_url="http://127.0.0.1:8000/v1",
        model_roles=("judge",),
        offline=True,
    )

    role_arg = commands[0][commands[0].index("--model-role") + 1]
    role_spec = json.loads(role_arg.removeprefix("judge="))
    assert role_spec == {
        "model": "vllm/local-model",
        "model_args": {"base_url": "http://127.0.0.1:8000/v1"},
    }
    assert result.status == "succeeded"


def test_accepted_error_message_handles_json_braces() -> None:
    error = subprocess.CalledProcessError(
        1,
        ["inspect", "eval"],
        output="require_optional_dependency",
    )
    reason = handle_accepted_errors(
        error,
        "fake_task",
        'fake descriptor {"model": "local"}',
    )
    assert "optional dependency error" in reason


# Tests for tools/run_evals.py credential handling.


@pytest.fixture
def no_kaggle_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the ~/.kaggle/kaggle.json fallback at a path that does not exist."""
    monkeypatch.setattr(
        os.path, "expanduser", lambda _p: str(tmp_path / "missing" / "kaggle.json")
    )


def test_has_kaggle_credentials_true_for_real_values(
    monkeypatch: pytest.MonkeyPatch, no_kaggle_json: None
) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "real-key")
    monkeypatch.setenv("KAGGLE_USERNAME", "real-user")
    assert _has_kaggle_credentials() is True


def test_has_kaggle_credentials_false_for_placeholder(
    monkeypatch: pytest.MonkeyPatch, no_kaggle_json: None
) -> None:
    # Placeholders let mlrc_bench load but are not real creds, so mle_bench
    # must still be treated as un-runnable and skipped.
    monkeypatch.setenv("KAGGLE_KEY", PLACEHOLDER_VALUE)
    monkeypatch.setenv("KAGGLE_USERNAME", PLACEHOLDER_VALUE)
    assert _has_kaggle_credentials() is False


def test_has_kaggle_credentials_false_when_placeholder_in_either_field(
    monkeypatch: pytest.MonkeyPatch, no_kaggle_json: None
) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "real-key")
    monkeypatch.setenv("KAGGLE_USERNAME", PLACEHOLDER_VALUE)
    assert _has_kaggle_credentials() is False


def test_has_kaggle_credentials_false_when_empty(
    monkeypatch: pytest.MonkeyPatch, no_kaggle_json: None
) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "")
    monkeypatch.setenv("KAGGLE_USERNAME", "")
    assert _has_kaggle_credentials() is False


def test_inject_placeholder_fills_empty_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # The smoke-test workflow sets these from secrets, which resolve to "".
    monkeypatch.setenv("KAGGLE_KEY", "")
    assert inject_placeholder_llm_keys()["KAGGLE_KEY"] == PLACEHOLDER_VALUE


def test_inject_placeholder_fills_unset_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    assert inject_placeholder_llm_keys()["KAGGLE_KEY"] == PLACEHOLDER_VALUE


def test_inject_placeholder_preserves_real_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real credential must not be overwritten with the placeholder (#1591).
    monkeypatch.setenv("KAGGLE_KEY", "real-key")
    assert "KAGGLE_KEY" not in inject_placeholder_llm_keys()
