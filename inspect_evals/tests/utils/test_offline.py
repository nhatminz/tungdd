import pytest
from inspect_ai.dataset import MemoryDataset, Sample

from inspect_evals.offline import (
    OFFLINE_BLOCK_MESSAGE,
    OfflineNetworkError,
    ensure_offline_address,
    is_local_network_host,
)
from inspect_evals.utils.offline import filter_offline_samples


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "::1", "localhost", "sandbox-service", "10.0.0.2"],
)
def test_local_network_hosts_are_allowed(host: str) -> None:
    assert is_local_network_host(host)
    ensure_offline_address((host, 8000))


@pytest.mark.parametrize("host", ["api.openai.com", "huggingface.co", "8.8.8.8"])
def test_public_network_hosts_are_blocked(host: str) -> None:
    assert not is_local_network_host(host)
    with pytest.raises(OfflineNetworkError, match=OFFLINE_BLOCK_MESSAGE):
        ensure_offline_address((host, 443))


def test_offline_only_filters_samples_marked_as_requiring_internet() -> None:
    dataset = MemoryDataset(
        [
            Sample(id="offline", input="offline", metadata={}),
            Sample(
                id="online",
                input="online",
                metadata={"requires_internet": True},
            ),
        ]
    )

    filtered = filter_offline_samples(dataset, offline_only=True)

    assert [sample.id for sample in filtered] == ["offline"]


def test_offline_only_false_preserves_dataset() -> None:
    dataset = MemoryDataset(
        [
            Sample(
                id="online",
                input="online",
                metadata={"requires_internet": True},
            )
        ]
    )

    assert filter_offline_samples(dataset, offline_only=False) is dataset


def test_offline_only_rejects_invalid_metadata() -> None:
    dataset = MemoryDataset(
        [
            Sample(
                id="invalid",
                input="invalid",
                metadata={"requires_internet": "yes"},
            )
        ]
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        filter_offline_samples(dataset, offline_only=True)
