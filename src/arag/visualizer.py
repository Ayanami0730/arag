"""Visualization tools for ARAG trajectory and performance analysis."""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryNode:
    """Single node in the agent trajectory."""

    loop: int
    step: int
    tool_name: str
    arguments: Dict[str, Any]
    result_preview: str
    tokens: int
    duration: float
    success: bool


class TrajectoryVisualizer:
    """Visualize agent execution trajectory."""

    def __init__(self, trajectory: List[Dict[str, Any]], perf_data: Dict[str, Any] = None):
        self.trajectory = trajectory
        self.perf_data = perf_data or {}

    def render_text(self) -> str:
        """Render trajectory as ASCII art."""
        if not self.trajectory:
            return "[Empty trajectory]"

        lines = []
        lines.append("=" * 70)
        lines.append("  AGENT EXECUTION TRAJECTORY")
        lines.append("=" * 70)

        prev_loop = 0
        for i, entry in enumerate(self.trajectory, 1):
            loop = entry.get("loop", 0)
            if loop != prev_loop:
                if prev_loop != 0:
                    lines.append("")
                lines.append(f"─── Loop {loop} ───")
                prev_loop = loop

            tool_name = entry.get("tool_name", "unknown")
            args = entry.get("arguments", {})
            result = entry.get("tool_result", "")
            tokens = entry.get("retrieved_tokens", 0)
            success = entry.get("success", True) if "success" in entry else True

            status = "OK" if success else "ERR"
            result_preview = result[:100] + "..." if len(result) > 100 else result
            result_preview = result_preview.replace("\n", " ")

            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""

            lines.append(f"  [{i:2d}] {tool_name:<20} {status:<4} | {args_str[:40]}")
            if tokens > 0:
                lines.append(f"       Tokens: {tokens:,} | Result: {result_preview[:50]}")
            else:
                lines.append(f"       Result: {result_preview[:60]}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def render_timeline(self) -> str:
        """Render timeline view of execution."""
        if not self.trajectory:
            return "[Empty trajectory]"

        lines = []
        lines.append("\n  TIMELINE VIEW")
        lines.append("-" * 50)

        loop_groups: Dict[int, List[Dict]] = {}
        for entry in self.trajectory:
            loop = entry.get("loop", 0)
            if loop not in loop_groups:
                loop_groups[loop] = []
            loop_groups[loop].append(entry)

        for loop, entries in sorted(loop_groups.items()):
            lines.append(f"\n  Loop {loop}:")
            for entry in entries:
                tool_name = entry.get("tool_name", "?")
                duration = entry.get("duration", 0)
                tokens = entry.get("retrieved_tokens", 0)
                bars = "█" * min(int(duration * 10), 40) if duration > 0 else ""
                lines.append(
                    f"    {tool_name:<18} {duration:5.2f}s {bars} "
                    f"{f'({tokens:,} tok)' if tokens > 0 else ''}"
                )

        return "\n".join(lines)

    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-tool statistics."""
        stats: Dict[str, Dict[str, Any]] = {}
        for entry in self.trajectory:
            tool_name = entry.get("tool_name", "unknown")
            if tool_name not in stats:
                stats[tool_name] = {"count": 0, "total_tokens": 0, "total_duration": 0.0}
            stats[tool_name]["count"] += 1
            stats[tool_name]["total_tokens"] += entry.get("retrieved_tokens", 0)
            stats[tool_name]["total_duration"] += entry.get("duration", 0)
        return stats

    def render_stats(self) -> str:
        """Render trajectory statistics."""
        stats = self.get_tool_stats()
        if not stats:
            return "[No statistics available]"

        lines = []
        lines.append("\n  TOOL USAGE STATISTICS")
        lines.append("-" * 50)
        lines.append(f"  {'Tool':<20} {'Calls':>6} {'Tokens':>10} {'Time(s)':>10}")
        lines.append("-" * 50)

        for tool_name, data in sorted(stats.items(), key=lambda x: -x[1]["total_tokens"]):
            lines.append(
                f"  {tool_name:<20} {data['count']:6d} "
                f"{data['total_tokens']:10,d} {data['total_duration']:10.2f}"
            )

        return "\n".join(lines)


class PerfDashboard:
    """Dashboard for performance data analysis."""

    def __init__(self, perf_data: Dict[str, Any]):
        self.perf_data = perf_data

    def render_summary(self) -> str:
        """Render performance summary."""
        summary = self.perf_data.get("summary", {})
        if not summary:
            return "[No performance data]"

        lines = []
        sep = "=" * 50

        lines.append(f"\n{sep}")
        lines.append("  PERFORMANCE SUMMARY")
        lines.append(sep)

        lines.append(
            f"  Duration: {summary.get('total_duration', 0):.2f}s | "
            f"LLM: {summary.get('llm_duration', 0):.2f}s | "
            f"Tools: {summary.get('tool_duration', 0):.2f}s"
        )
        lines.append(
            f"  LLM Calls: {summary.get('llm_call_count', 0)} | "
            f"Tool Calls: {summary.get('tool_call_count', 0)} | "
            f"Cost: ${summary.get('total_cost', 0):.6f}"
        )
        lines.append(
            f"  Tokens - Input: {summary.get('total_input_tokens', 0):,} | "
            f"Output: {summary.get('total_output_tokens', 0):,} | "
            f"Retrieved: {summary.get('total_retrieved_tokens', 0):,}"
        )

        return "\n".join(lines)

    def render_timing_breakdown(self) -> str:
        """Render detailed timing breakdown."""
        summary = self.perf_data.get("summary", {})
        if not summary:
            return "[No performance data]"

        llm_dist = summary.get("llm_duration_distribution", {})
        tool_dist = summary.get("tool_duration_distribution", {})

        lines = []
        lines.append(f"\n  LLM CALL DURATIONS")
        lines.append("-" * 40)
        lines.append(
            f"  Mean: {llm_dist.get('mean', 0):.3f}s | "
            f"P50: {llm_dist.get('p50', 0):.3f}s | "
            f"P90: {llm_dist.get('p90', 0):.3f}s"
        )

        lines.append(f"\n  TOOL CALL DURATIONS")
        lines.append("-" * 40)
        lines.append(
            f"  Mean: {tool_dist.get('mean', 0):.3f}s | "
            f"P50: {tool_dist.get('p50', 0):.3f}s | "
            f"P90: {tool_dist.get('p90', 0):.3f}s"
        )

        return "\n".join(lines)

    def render_context_tokens(self) -> str:
        """Render context token distribution across loops."""
        records = self.perf_data.get("summary", {}).get("context_token_records", [])
        if not records:
            return "[No context token data]"

        lines = []
        lines.append(f"\n  CONTEXT TOKEN DISTRIBUTION")
        lines.append("-" * 60)
        lines.append(f"  {'Loop':>4}  {'Total':>8}  {'System':>8}  {'History':>8}  {'Latest':>8}")
        lines.append("-" * 60)

        for rec in records:
            lines.append(
                f"  {rec.get('loop', 0):4d}  "
                f"{rec.get('message_tokens', 0):8,d}  "
                f"{rec.get('system_prompt_tokens', 0):8,d}  "
                f"{rec.get('history_tokens', 0):8,d}  "
                f"{rec.get('latest_tool_tokens', 0):8,d}"
            )

        return "\n".join(lines)


def load_results(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def visualize_result(result: Dict[str, Any]) -> str:
    """Generate complete visualization for a single result."""
    trajectory = result.get("trajectory", [])
    perf_data = result.get("perf", {})

    viz = TrajectoryVisualizer(trajectory, perf_data)
    dashboard = PerfDashboard(perf_data)

    parts = [
        viz.render_text(),
        viz.render_timeline(),
        viz.render_stats(),
        dashboard.render_summary(),
        dashboard.render_timing_breakdown(),
        dashboard.render_context_tokens(),
    ]

    return "\n".join(filter(None, parts))


def compare_results(results: List[Dict[str, Any]], labels: List[str] = None) -> str:
    """Compare multiple results side by side."""
    if not results:
        return "[No results to compare]"

    labels = labels or [f"Result {i + 1}" for i in range(len(results))]

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  COMPARISON VIEW")
    lines.append("=" * 70)
    lines.append(f"  {'Metric':<25} " + " ".join(f"{l:<15}" for l in labels))
    lines.append("-" * 70)

    metrics = []
    for r in results:
        perf = r.get("perf", {}).get("summary", {})
        metrics.append(
            {
                "duration": perf.get("total_duration", 0),
                "llm_calls": perf.get("llm_call_count", 0),
                "tool_calls": perf.get("tool_call_count", 0),
                "cost": perf.get("total_cost", 0),
                "input_tokens": perf.get("total_input_tokens", 0),
                "output_tokens": perf.get("total_output_tokens", 0),
                "retrieved_tokens": perf.get("total_retrieved_tokens", 0),
            }
        )

    lines.append(f"  {'Duration (s)':<25} " + " ".join(f"{m['duration']:<15.2f}" for m in metrics))
    lines.append(f"  {'LLM Calls':<25} " + " ".join(f"{m['llm_calls']:<15d}" for m in metrics))
    lines.append(f"  {'Tool Calls':<25} " + " ".join(f"{m['tool_calls']:<15d}" for m in metrics))
    lines.append(f"  {'Cost ($)':<25} " + " ".join(f"{m['cost']:<15.6f}" for m in metrics))
    lines.append(
        f"  {'Input Tokens':<25} " + " ".join(f"{m['input_tokens']:<15,}" for m in metrics)
    )
    lines.append(
        f"  {'Output Tokens':<25} " + " ".join(f"{m['output_tokens']:<15,}" for m in metrics)
    )
    lines.append(
        f"  {'Retrieved Tokens':<25} " + " ".join(f"{m['retrieved_tokens']:<15,}" for m in metrics)
    )

    lines.append("=" * 70)
    return "\n".join(lines)
