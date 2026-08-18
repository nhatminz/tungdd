"""Helpers for filtering mixed online/offline datasets."""

from __future__ import annotations

import logging

from inspect_ai.dataset import Dataset, Sample

REQUIRES_INTERNET_METADATA_KEY = "requires_internet"

logger = logging.getLogger(__name__)


def sample_requires_internet(sample: Sample) -> bool:
    """Return a sample's declared Internet requirement.

    Dataset adapters opt in by setting ``metadata.requires_internet`` to a
    boolean. Missing metadata is treated as offline-compatible.
    """
    value = sample.metadata.get(REQUIRES_INTERNET_METADATA_KEY, False)
    if not isinstance(value, bool):
        raise ValueError(
            f"Sample {sample.id!r} metadata.{REQUIRES_INTERNET_METADATA_KEY} "
            f"must be a boolean, got {value!r}"
        )
    return value


def filter_offline_samples(dataset: Dataset, *, offline_only: bool) -> Dataset:
    """Filter samples declaring an Internet requirement when requested."""
    if not offline_only:
        return dataset

    filtered = dataset.filter(lambda sample: not sample_requires_internet(sample))
    logger.info(
        "offline_only excluded %d of %d Internet-dependent sample(s)",
        len(dataset) - len(filtered),
        len(dataset),
    )
    return filtered
