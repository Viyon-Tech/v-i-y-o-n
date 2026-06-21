"""NEXUS agent (data & analytics): CSV/Excel/JSON profiling, charts, and reports.

NEXUS runs pandas/matplotlib in a subprocess via tools.terminal (gated for
safety — running code over a data file is a side effect). It profiles datasets,
renders charts to PNG artifacts, and writes markdown reports. For a full
interactive dashboard it hands off to NOVA.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from agents.base_agent import AgentResult, BaseAgent
from tools import terminal

logger = logging.getLogger("viyon.nexus")

# Subprocess scripts (run via the configured Python so pandas is on the path).
_PROFILE_SCRIPT = r"""
import sys, json
import pandas as pd
p = sys.argv[1]
if p.endswith(('.xlsx', '.xls')):
    df = pd.read_excel(p)
elif p.endswith('.json'):
    df = pd.read_json(p)
else:
    df = pd.read_csv(p)
out = {"rows": int(len(df)), "cols": [str(c) for c in df.columns], "columns": {}}
for c in df.columns:
    s = df[c]
    info = {"dtype": str(s.dtype), "count": int(s.count()), "nulls": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s) and s.count():
        info.update({"mean": round(float(s.mean()), 4),
                     "min": float(s.min()), "max": float(s.max())})
    else:
        info["unique"] = int(s.nunique())
    out["columns"][str(c)] = info
print(json.dumps(out))
"""

_CHART_SCRIPT = r"""
import sys, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p, out_png, spec_json = sys.argv[1], sys.argv[2], sys.argv[3]
spec = json.loads(spec_json)
df = pd.read_csv(p) if p.endswith('.csv') else (
    pd.read_json(p) if p.endswith('.json') else pd.read_excel(p))
kind = spec.get("kind", "line")
x, y = spec.get("x"), spec.get("y")
ax = (df.plot(x=x, y=y, kind=kind) if x or y else df.plot(kind=kind))
plt.tight_layout()
plt.savefig(out_png)
print(out_png)
"""

_CHARTS_DIR = Path("~/.viyon/charts").expanduser()


class NexusAgent(BaseAgent):
    """VIYON's data & analytics agent."""

    name = "NEXUS"
    emoji = "📊"
    scope = "Data & analytics — CSV/Excel/JSON profiling, charts, reports."

    def system_prompt(self) -> str:
        return (
            "You are NEXUS 📊, VIYON's data analyst. You profile datasets, build charts, "
            "and write clear reports. State sample sizes and caveats; don't overclaim from "
            "small or messy data."
        )

    # -- capabilities ------------------------------------------------------

    async def profile(self, file: str) -> AgentResult:
        """Profile a data file: per-column dtype, counts, nulls, and numeric stats."""
        code, out, err = await terminal.run(
            [sys.executable, "-c", _PROFILE_SCRIPT, str(file)],
            approval_required=True,
            approval=self.approval,
            risk="low",
        )
        if code in (125, 126):
            return self.fail("Profiling wasn't approved.")
        if code != 0:
            return self.fail(f"Couldn't profile {Path(file).name}: {err[:200].strip()}")
        try:
            prof = json.loads(out)
        except json.JSONDecodeError:
            return self.fail("Couldn't parse the profile output.")

        lines = [f"{prof['rows']} rows × {len(prof['cols'])} columns"]
        for col, info in prof["columns"].items():
            if "mean" in info:
                extra = f"mean={info['mean']} min={info['min']} max={info['max']}"
            else:
                extra = f"unique={info.get('unique')}"
            lines.append(
                f"  {col} ({info['dtype']}): count={info['count']} nulls={info['nulls']} {extra}"
            )
        summary = (
            f"{Path(file).name}: {prof['rows']} rows, {len(prof['cols'])} columns "
            f"({', '.join(prof['cols'][:6])})."
        )
        return self.succeed(summary, detail="\n".join(lines))

    async def chart(self, file: str, spec: dict) -> AgentResult:
        """Render a chart from a data file and save a PNG to artifacts."""
        _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        out_png = str(_CHARTS_DIR / f"{Path(file).stem}-{spec.get('kind', 'line')}.png")
        code, out, err = await terminal.run(
            [sys.executable, "-c", _CHART_SCRIPT, str(file), out_png, json.dumps(spec)],
            approval_required=True,
            approval=self.approval,
            risk="low",
        )
        if code in (125, 126):
            return self.fail("Charting wasn't approved.")
        if code != 0:
            hint = " (install matplotlib)" if "matplotlib" in err else ""
            return self.fail(f"Couldn't build the chart{hint}: {err[:200].strip()}")
        return self.succeed(f"Saved a {spec.get('kind', 'line')} chart.", artifacts=[out_png])

    async def report(self, file: str, ctx: dict | None = None) -> AgentResult:
        """Profile a file then ask the LLM to write a markdown summary."""
        prof = await self.profile(file)
        if not prof.ok:
            return prof
        narrative = await self.think(
            f"Write a short markdown data report for {Path(file).name} based on this "
            f"profile:\n{prof.detail}",
            ctx,
        )
        markdown = narrative or f"# {Path(file).name}\n\n{prof.detail}"
        return self.succeed(f"Wrote a report on {Path(file).name}.", detail=markdown)

    # -- run ---------------------------------------------------------------

    async def run(self, task: str, ctx: dict) -> AgentResult:
        """Dispatch a data request; hand off to NOVA for a full dashboard."""
        low = (task or "").lower()
        file = self._extract_file(task) or (ctx or {}).get("file")

        if "dashboard" in low:
            return AgentResult(
                agent=self.name,
                ok=True,
                summary="A full dashboard is a build job — handing off to NOVA.",
                detail=task,
                handoff="NOVA",
            )

        if not file:
            return self.fail("Point me at a data file (.csv, .xlsx, or .json) to analyze.")

        if "chart" in low or "plot" in low or "graph" in low:
            return await self.chart(file, {"kind": "line"})
        if "report" in low:
            return await self.report(file, ctx)
        return await self.profile(file)

    @staticmethod
    def _extract_file(task: str) -> str | None:
        m = re.search(r"(\S+\.(?:csv|xlsx|xls|json))", task or "", re.IGNORECASE)
        return m.group(1) if m else None
