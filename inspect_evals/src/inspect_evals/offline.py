"""Runtime network guard for offline evaluation.

The guard is deliberately opt-in through ``INSPECT_EVALS_OFFLINE``. It allows
loopback, private, link-local, and single-label hosts so local model servers and
sandbox services continue to work, while blocking public Internet hosts.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any

OFFLINE_ENV_VAR = "INSPECT_EVALS_OFFLINE"
OFFLINE_BLOCK_MESSAGE = "offline mode blocked outbound network access"

logger = logging.getLogger(__name__)

_guard_state = {"installed": False}


class OfflineNetworkError(ConnectionError):
    """Raised when an offline evaluation attempts public network access."""


def is_local_network_host(host: object) -> bool:
    """Return whether a host is safe to access during an offline run."""
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if not isinstance(host, str):
        return False

    host = host.strip("[]").rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Single-label names are commonly Docker/Podman service names. Public
        # Internet names contain a dot and are rejected before DNS resolution.
        return bool(host) and "." not in host

    return address.is_loopback or address.is_private or address.is_link_local


def ensure_offline_address(address: object) -> None:
    """Raise if a socket address targets a public network host."""
    if isinstance(address, str):
        return  # Unix-domain socket path.
    if not isinstance(address, tuple) or not address:
        raise OfflineNetworkError(f"{OFFLINE_BLOCK_MESSAGE}: {address!r}")

    host = address[0]
    if not is_local_network_host(host):
        raise OfflineNetworkError(f"{OFFLINE_BLOCK_MESSAGE}: {host!r}")


def install_offline_network_guard() -> bool:
    """Install the socket guard when ``INSPECT_EVALS_OFFLINE`` is enabled."""
    if _guard_state["installed"] or os.environ.get(OFFLINE_ENV_VAR) != "1":
        return False

    original_socket = socket.socket
    original_getaddrinfo = socket.getaddrinfo

    class OfflineSocket(original_socket):  # type: ignore[misc, valid-type]
        def connect(self, address: Any) -> None:
            ensure_offline_address(address)
            return super().connect(address)

        def connect_ex(self, address: Any) -> int:
            ensure_offline_address(address)
            return super().connect_ex(address)

    def offline_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and not is_local_network_host(host):
            raise OfflineNetworkError(f"{OFFLINE_BLOCK_MESSAGE}: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket = OfflineSocket
    socket.getaddrinfo = offline_getaddrinfo
    _guard_state["installed"] = True
    logger.info("Inspect Evals offline network guard enabled")
    return True
