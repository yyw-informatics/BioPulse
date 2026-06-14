"""Network policy guard for the agent's run_python subprocess.

Ported from BioPulse (``biopulse/agent/netguard.py``). The guard monkeypatches the socket layer in the
child process to enforce the run's policy and append every connection attempt to an audit log:

  - ``block_all`` (L1): every outbound connection raises.
  - ``blacklist`` (L2/L3/L4): connections to answer-revealing hosts raise, others are allowed. Every
    attempt (allowed or blocked) is logged, so a blocked reach to an answer source is recorded.

Enforcement is at the Python socket layer (covers ``requests`` / ``urllib`` / ``httpx``), not a
subprocess escape; the audit log verifies nothing slipped through. :func:`install_netguard` returns the
``(env, cmd_prefix)`` that make ``execute_python`` run the agent's script under the guard; with no policy
it returns ``(None, [])`` and behavior is unchanged.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class NetPolicy:
    """Network policy: the mode and, for blacklist mode, the answer-revealing host tokens."""

    mode: str  # "block_all" | "blacklist"
    blacklist: tuple[str, ...] = ()


def _is_ip_literal(host: object) -> bool:
    """Return True if ``host`` is a numeric IP literal (IPv4 or IPv6). Such connects skip DNS, so they
    bypass the getaddrinfo gate and must be checked at connect time."""
    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def is_blocked(host: object, mode: str, blacklist: Iterable[str]) -> bool:
    """Return whether a connection to ``host`` should be blocked. Pure, no I/O."""
    if mode == "block_all":
        return True
    if mode != "blacklist":
        return False
    h = str(host).lower()
    for token in blacklist:
        t = str(token).lower().strip()
        if t and (h == t or h.endswith("." + t) or t in h):
            return True
    return False


def install(mode: str, blacklist: Iterable[str] = (), log_path: Optional[str] = None) -> None:
    """Monkeypatch the socket layer to enforce ``mode`` and append every attempt to ``log_path``."""
    blacklist = tuple(blacklist)

    def _log(host: object, port: object, blocked: bool) -> None:
        if not log_path:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"host": str(host), "port": port, "blocked": bool(blocked), "mode": mode}) + "\n")
        except OSError:
            pass

    original_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        blocked = is_blocked(host, mode, blacklist)
        _log(host, port, blocked)
        if blocked:
            raise OSError(f"biopulse netguard: network access to {host!r} blocked (policy={mode})")
        return original_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]

    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, (tuple, list)) and address else address
        is_ip = _is_ip_literal(host)
        # Hostname connects are gated by getaddrinfo; gate direct-IP connects (no DNS) here, and gate
        # everything under block_all.
        if mode == "block_all" or is_ip:
            blocked = is_blocked(host, mode, blacklist)
            if is_ip:
                port = address[1] if isinstance(address, (tuple, list)) and len(address) > 1 else None
                _log(host, port, blocked)
            if blocked:
                raise OSError(f"biopulse netguard: connection to {host!r} blocked (policy={mode})")
        return original_connect(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[assignment]


def install_from_env() -> None:
    """Install the guard from ``BIOPULSE_NET_*`` env vars (set by :func:`install_netguard`). No-op when off."""
    mode = os.environ.get("BIOPULSE_NET_MODE", "off")
    if mode in ("", "off"):
        return
    blacklist = [host for host in os.environ.get("BIOPULSE_NET_BLACKLIST", "").split(",") if host]
    install(mode, blacklist, os.environ.get("BIOPULSE_NET_LOG") or None)


# Child-process prelude: install the guard from the env, then run the agent's script as __main__ with a
# clean argv. Invoked as `python -c <_NETGUARD_BOOT> <script>`.
_NETGUARD_BOOT = (
    "import sys, runpy, biopulse_lg.netguard as _ng; "
    "_ng.install_from_env(); "
    "_p = sys.argv[1]; sys.argv = [_p]; "
    "runpy.run_path(_p, run_name='__main__')"
)


def install_netguard(policy: Optional[NetPolicy], log_path=None) -> tuple[Optional[dict], list[str]]:
    """Return ``(env, cmd_prefix)`` to run a script under ``policy``.

    With ``policy is None`` returns ``(None, [])``; the caller then runs ``[python, script]`` with the
    inherited environment. Otherwise returns an environment carrying ``BIOPULSE_NET_*`` and the
    ``-c <boot>`` prefix that installs the guard before the script runs.
    """
    if policy is None:
        return None, []
    env = dict(os.environ)
    env["BIOPULSE_NET_MODE"] = policy.mode
    env["BIOPULSE_NET_BLACKLIST"] = ",".join(policy.blacklist)
    if log_path is not None:
        env["BIOPULSE_NET_LOG"] = str(log_path)
    return env, ["-c", _NETGUARD_BOOT]
