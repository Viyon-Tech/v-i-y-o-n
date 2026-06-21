"""Approval Gate: every destructive or state-changing action requires explicit user 'yes'.

The gate is front-end agnostic: it asks for confirmation through a pluggable
callback. The default callback reads from the console (:func:`console_callback`);
a voice front-end can supply an (optionally async) callback that returns the
user's spoken yes/no. Only an explicit affirmative approves — everything else,
including silence, is treated as an abort.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable

from logs.logger import VLogger, log

# A front-end takes a prompt string and returns the user's answer. The return
# value may be a plain str or an awaitable resolving to one (for voice mode).
Callback = Callable[[str], "str | Awaitable[str]"]

RISK_LEVELS = ("low", "medium", "high")

# Strings (case-insensitive, stripped) that count as explicit approval.
AFFIRMATIVES = frozenset({"yes", "y", "go ahead", "do it", "confirm"})

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def console_callback(prompt: str) -> str:
    """Default front-end: ask for confirmation on the console."""
    return input(prompt)


def _load_safety_config() -> dict:
    """Load the ``safety`` section of config.yaml (empty dict if unavailable)."""
    if not _CONFIG_PATH.exists():
        return {}
    import yaml  # local import: keeps PyYAML off the hot path for injected configs

    data = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    return data.get("safety", {}) or {}


class ApprovalGate:
    """Gates destructive/state-changing actions behind an explicit confirmation.

    Args:
        callback: Front-end that prompts the user. Defaults to the console.
        config: The ``safety`` config mapping (``auto_approve_low``,
            ``require_confirm_for``). Loaded from config.yaml when omitted.
        logger: Logger for recording decisions. Defaults to the shared singleton.
        mode: Human-readable label for the active front-end (``"console"`` /
            ``"voice"``); recorded with each decision.
    """

    def __init__(
        self,
        callback: Callback | None = None,
        config: dict | None = None,
        logger: VLogger | None = None,
        mode: str = "console",
    ) -> None:
        self._callback: Callback = callback or console_callback
        self._config = config if config is not None else _load_safety_config()
        self._log = logger if logger is not None else log
        self.mode = mode

    async def request(self, action: str, detail: str, risk: str) -> bool:
        """Ask the user to approve ``action``; return True only on explicit yes.

        Args:
            action: Short action identifier (e.g. ``"delete"``, ``"git_push"``).
            detail: Human-readable description of exactly what will happen.
            risk: One of ``"low"``, ``"medium"``, ``"high"``.

        Raises:
            ValueError: if ``risk`` is not a recognized level.
        """
        risk = (risk or "").lower()
        if risk not in RISK_LEVELS:
            raise ValueError(f"risk must be one of {RISK_LEVELS}, got {risk!r}")

        require_confirm_for = self._config.get("require_confirm_for") or []
        always_ask = action in require_confirm_for
        auto_approve_low = bool(self._config.get("auto_approve_low", False))

        # Low risk may auto-approve only when enabled AND not force-listed.
        if risk == "low" and auto_approve_low and not always_ask:
            self._record(action, detail, risk, approved=True, mode="auto")
            return True

        answer = await self._ask(action, detail, risk)
        approved = self._is_affirmative(answer)
        self._record(action, detail, risk, approved=approved, mode=self.mode)
        return approved

    async def _ask(self, action: str, detail: str, risk: str) -> str:
        """Invoke the front-end callback (awaiting it if async) and return the answer."""
        prompt = self._build_prompt(action, detail, risk)
        result: Any = self._callback(prompt)
        if inspect.isawaitable(result):
            result = await result
        return "" if result is None else str(result)

    @staticmethod
    def _build_prompt(action: str, detail: str, risk: str) -> str:
        """Format the confirmation prompt shown to the user."""
        return (
            f"\n[VIYON APPROVAL] action={action!r}  risk={risk.upper()}\n"
            f"  {detail}\n"
            "Proceed? (yes/no): "
        )

    @staticmethod
    def _is_affirmative(answer: str) -> bool:
        """True only if the answer is an explicit affirmative."""
        return str(answer).strip().lower() in AFFIRMATIVES

    def _record(
        self, action: str, detail: str, risk: str, *, approved: bool, mode: str
    ) -> None:
        """Log the approval decision via the structured logger."""
        decision = "approved" if approved else "denied"
        self._log.log_command(
            {
                "raw_input": action,
                "parsed_intent": {"approval": action, "risk": risk, "detail": detail},
                "agents": [],
                "steps": [{"approval": action, "risk": risk, "via": mode, "decision": decision}],
                "result": decision,
                "confirmed": approved,
                "status": "ok" if approved else "aborted",
            }
        )
