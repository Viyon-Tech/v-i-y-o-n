"""GHOST agent (system monitor): CPU, memory, disk, battery, and processes via psutil.

GHOST reports system health in a spoken-friendly summary and packs the numbers
into ``AgentResult.detail``. Killing a process is gated at high risk through the
Approval Gate. psutil is imported lazily so the module loads without it.
"""

from __future__ import annotations

import asyncio
import logging
import re

from agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger("viyon.ghost")


class GhostAgent(BaseAgent):
    """Reports system metrics and (with approval) terminates processes."""

    name = "GHOST"
    emoji = "👁️"
    scope = "System monitor — CPU, memory, disk, battery, processes."

    def system_prompt(self) -> str:
        return (
            "You are GHOST 👁️, VIYON's system monitor. Report CPU, memory, disk, "
            "battery, and processes plainly and concisely. You never kill a process "
            "without explicit confirmation."
        )

    # -- metrics -----------------------------------------------------------

    async def snapshot(self) -> dict:
        """Return a one-shot snapshot of CPU/memory/disk/battery."""
        import psutil

        cpu = await asyncio.to_thread(psutil.cpu_percent, 0.3)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        battery = psutil.sensors_battery()
        return {
            "cpu_percent": cpu,
            "mem_percent": mem.percent,
            "mem_used_gb": round(mem.used / 1e9, 1),
            "mem_total_gb": round(mem.total / 1e9, 1),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / 1e9, 1),
            "battery_percent": round(battery.percent) if battery else None,
            "battery_plugged": battery.power_plugged if battery else None,
        }

    async def top_processes(self, by: str = "cpu", n: int = 5) -> list[dict]:
        """Return the top ``n`` processes by ``cpu`` or ``mem``."""
        import psutil

        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            info = proc.info
            procs.append(
                {
                    "pid": info["pid"],
                    "name": info["name"] or "?",
                    "cpu": round(info.get("cpu_percent") or 0.0, 1),
                    "mem": round(info.get("memory_percent") or 0.0, 1),
                }
            )
        key = "mem" if by == "mem" else "cpu"
        procs.sort(key=lambda p: p[key], reverse=True)
        return procs[:n]

    async def kill_process(self, pid: int) -> AgentResult:
        """Terminate a process by pid — gated at HIGH risk."""
        import psutil

        try:
            proc = psutil.Process(pid)
            pname = proc.name()
        except Exception as exc:
            return self.fail(f"No process with pid {pid}.", detail=str(exc))

        async def _terminate() -> AgentResult:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 3)
            except Exception:
                proc.kill()
            return self.succeed(f"Terminated {pname} (pid {pid}).")

        return await self.guarded(
            "kill_process", f"kill {pname} (pid {pid})", "high", _terminate
        )

    async def monitor(self, seconds: int = 5) -> dict:
        """Sample CPU once a second for ``seconds`` and flag spikes (>85%)."""
        import psutil

        samples: list[float] = []
        for _ in range(max(1, int(seconds))):
            samples.append(psutil.cpu_percent(interval=None))
            await asyncio.sleep(1)
        anomalies = [round(v, 1) for v in samples if v > 85]
        return {
            "avg": round(sum(samples) / len(samples), 1),
            "peak": round(max(samples), 1),
            "samples": [round(v, 1) for v in samples],
            "anomalies": anomalies,
        }

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Interpret a monitoring request and return a spoken summary + detail."""
        t = (task or "").lower()

        kill_match = re.search(r"kill .*?\b(\d{2,7})\b", t)
        if kill_match:
            return await self.kill_process(int(kill_match.group(1)))

        if "monitor" in t or "watch" in t:
            secs_match = re.search(r"(\d+)\s*(?:second|sec|s)\b", t)
            secs = int(secs_match.group(1)) if secs_match else 5
            stats = await self.monitor(secs)
            note = (
                f"{len(stats['anomalies'])} CPU spike(s) over 85%."
                if stats["anomalies"]
                else "no anomalies."
            )
            summary = (
                f"Monitored {secs}s: average CPU {stats['avg']}%, peak {stats['peak']}%, {note}"
            )
            detail = f"samples: {stats['samples']}\nanomalies: {stats['anomalies']}"
            return self.succeed(summary, detail=detail)

        # Default: a health snapshot, with top processes if asked.
        snap = await self.snapshot()
        bat = (
            f", battery {snap['battery_percent']}%"
            f"{' charging' if snap['battery_plugged'] else ''}"
            if snap["battery_percent"] is not None
            else ""
        )
        summary = (
            f"CPU {snap['cpu_percent']}%, memory {snap['mem_percent']}% used, "
            f"disk {snap['disk_percent']}% full{bat}."
        )
        detail_lines = [
            f"CPU:     {snap['cpu_percent']}%",
            f"Memory:  {snap['mem_percent']}% ({snap['mem_used_gb']}/{snap['mem_total_gb']} GB)",
            f"Disk:    {snap['disk_percent']}% used, {snap['disk_free_gb']} GB free",
            f"Battery: {snap['battery_percent']}%" if snap["battery_percent"] is not None
            else "Battery: n/a",
        ]

        if "process" in t or "top" in t or "using" in t:
            by = "mem" if ("mem" in t or "memory" in t or "ram" in t) else "cpu"
            top = await self.top_processes(by=by, n=5)
            detail_lines.append(f"\nTop processes by {by}:")
            detail_lines += [
                f"  {p['pid']:>7}  {p['name'][:25]:<25}  cpu {p['cpu']}%  mem {p['mem']}%"
                for p in top
            ]
            if top:
                summary += f" Top {by}: {top[0]['name']} ({top[0][by]}%)."

        return self.succeed(summary, detail="\n".join(detail_lines))
