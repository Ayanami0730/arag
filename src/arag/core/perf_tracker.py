"""Performance tracker for ARAG query and tool call profiling.

记录每次 query 的耗时、token 消耗和工具调用情况，
并在结束时输出结构化性能报告，为效率优化提供情报支持。
"""

import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("arag.perf")


@dataclass
class ContextTokenRecord:
    """单轮上下文 token 分布记录。"""

    loop: int
    message_tokens: int
    system_prompt_tokens: int
    history_tokens: int
    latest_tool_tokens: int


@dataclass
class LLMCallRecord:
    """单次 LLM 调用的性能记录。"""

    loop: int
    duration: float
    input_tokens: int
    output_tokens: int
    cost: float
    has_tool_calls: bool
    forced_final: bool = False
    phase_durations: Optional[Dict[str, float]] = None


@dataclass
class ToolCallRecord:
    """单次工具调用的性能记录。"""

    loop: int
    tool_name: str
    duration: float
    retrieved_tokens: int
    success: bool
    arguments: Dict[str, Any] = field(default_factory=dict)


class QueryPerfTracker:
    """单次 query 的完整性能追踪器。

    在 BaseAgent.run() 中创建，收集所有 LLM 调用和工具调用的
    耗时与 token 数据，最终输出格式化报告。
    """

    def __init__(self, query: str, model: str = ""):
        self.query = query
        self.model = model
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.llm_calls: List[LLMCallRecord] = []
        self.tool_calls: List[ToolCallRecord] = []
        self.context_token_records: List[ContextTokenRecord] = []

    def record_llm_call(
        self,
        loop: int,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        has_tool_calls: bool,
        forced_final: bool = False,
        phase_durations: Optional[Dict[str, float]] = None,
    ):
        self.llm_calls.append(
            LLMCallRecord(
                loop=loop,
                duration=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                has_tool_calls=has_tool_calls,
                forced_final=forced_final,
                phase_durations=phase_durations,
            )
        )

    def record_tool_call(
        self,
        loop: int,
        tool_name: str,
        duration: float,
        retrieved_tokens: int,
        success: bool,
        arguments: Optional[Dict[str, Any]] = None,
    ):
        self.tool_calls.append(
            ToolCallRecord(
                loop=loop,
                tool_name=tool_name,
                duration=duration,
                retrieved_tokens=retrieved_tokens,
                success=success,
                arguments=arguments or {},
            )
        )

    def record_context_tokens(
        self,
        loop: int,
        message_tokens: int,
        system_prompt_tokens: int,
        history_tokens: int,
        latest_tool_tokens: int,
    ):
        self.context_token_records.append(
            ContextTokenRecord(
                loop=loop,
                message_tokens=message_tokens,
                system_prompt_tokens=system_prompt_tokens,
                history_tokens=history_tokens,
                latest_tool_tokens=latest_tool_tokens,
            )
        )

    def finish(self):
        self.end_time = time.time()

    @property
    def total_duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def total_llm_duration(self) -> float:
        return sum(r.duration for r in self.llm_calls)

    @property
    def total_tool_duration(self) -> float:
        return sum(r.duration for r in self.tool_calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.llm_calls)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.llm_calls)

    @property
    def total_retrieved_tokens(self) -> int:
        return sum(r.retrieved_tokens for r in self.tool_calls)

    def _tool_distribution(self) -> Dict[str, Dict[str, Any]]:
        dist: Dict[str, Dict[str, Any]] = {}
        for r in self.tool_calls:
            if r.tool_name not in dist:
                dist[r.tool_name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "total_tokens": 0,
                    "errors": 0,
                }
            d = dist[r.tool_name]
            d["count"] += 1
            d["total_duration"] += r.duration
            d["total_tokens"] += r.retrieved_tokens
            if not r.success:
                d["errors"] += 1
        return dist

    @staticmethod
    def _distribution(values: List[float], digits: int = 3) -> Dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "sum": 0,
                "mean": 0,
                "min": 0,
                "p50": 0,
                "p90": 0,
                "max": 0,
            }

        sorted_values = sorted(values)

        def pick(p: float) -> float:
            if len(sorted_values) == 1:
                return sorted_values[0]
            idx = (len(sorted_values) - 1) * p
            lo = int(idx)
            hi = min(lo + 1, len(sorted_values) - 1)
            w = idx - lo
            return sorted_values[lo] * (1 - w) + sorted_values[hi] * w

        total = sum(values)
        return {
            "count": len(values),
            "sum": round(total, digits),
            "mean": round(total / len(values), digits),
            "min": round(min(values), digits),
            "p50": round(pick(0.5), digits),
            "p90": round(pick(0.9), digits),
            "max": round(max(values), digits),
        }

    def format_report(self) -> str:
        sep = "=" * 68
        thin = "-" * 68
        total_dur = self.total_duration
        llm_dur = self.total_llm_duration
        tool_dur = self.total_tool_duration
        other_dur = max(total_dur - llm_dur - tool_dur, 0)

        lines: List[str] = []
        lines.append(f"\n{sep}")
        lines.append("  ARAG Performance Report")
        lines.append(sep)

        query_preview = self.query[:80] + "..." if len(self.query) > 80 else self.query
        lines.append(f"  Query  : {query_preview}")
        if self.model:
            lines.append(f"  Model  : {self.model}")
        lines.append(
            f"  Total  : {total_dur:.2f}s | "
            f"LLM calls: {len(self.llm_calls)} | "
            f"Tool calls: {len(self.tool_calls)} | "
            f"Cost: ${self.total_cost:.6f}"
        )

        lines.append(f"\n{thin}")
        lines.append("  Timing Breakdown")
        lines.append(thin)
        pct = lambda v: f"{v / total_dur * 100:.1f}%" if total_dur > 0 else "0.0%"
        avg_llm = llm_dur / len(self.llm_calls) if self.llm_calls else 0
        avg_tool = tool_dur / len(self.tool_calls) if self.tool_calls else 0
        lines.append(
            f"  LLM    : {llm_dur:7.2f}s ({pct(llm_dur):>6}) | "
            f"{len(self.llm_calls)} calls | avg {avg_llm:.2f}s"
        )
        lines.append(
            f"  Tools  : {tool_dur:7.2f}s ({pct(tool_dur):>6}) | "
            f"{len(self.tool_calls)} calls | avg {avg_tool:.2f}s"
        )
        lines.append(f"  Other  : {other_dur:7.2f}s ({pct(other_dur):>6}) | overhead")

        if self.llm_calls:
            lines.append(f"\n{thin}")
            lines.append("  LLM Call Details")
            lines.append(thin)
            lines.append(
                f"  {'#':>3}  {'Loop':>4}  {'Duration':>8}  "
                f"{'In Tok':>8}  {'Out Tok':>8}  {'Cost':>10}  {'Tools?'}  {'Forced?'}"
            )
            for i, r in enumerate(self.llm_calls, 1):
                forced_mark = "FORCED" if r.forced_final else "No"
                lines.append(
                    f"  {i:3d}  {r.loop:4d}  {r.duration:7.2f}s  "
                    f"{r.input_tokens:8d}  {r.output_tokens:8d}  "
                    f"${r.cost:9.6f}  {'Yes' if r.has_tool_calls else 'No':>6}  {forced_mark}"
                )

        if self.tool_calls:
            lines.append(f"\n{thin}")
            lines.append("  Tool Call Details")
            lines.append(thin)
            lines.append(
                f"  {'#':>3}  {'Loop':>4}  {'Tool':<20}  "
                f"{'Duration':>8}  {'Ret Tok':>8}  {'Status'}"
            )
            for i, r in enumerate(self.tool_calls, 1):
                status = "OK" if r.success else "ERR"
                lines.append(
                    f"  {i:3d}  {r.loop:4d}  {r.tool_name:<20}  "
                    f"{r.duration:7.2f}s  {r.retrieved_tokens:8d}  {status}"
                )

        dist = self._tool_distribution()
        if dist:
            lines.append(f"\n{thin}")
            lines.append("  Tool Distribution")
            lines.append(thin)
            lines.append(
                f"  {'Tool':<20}  {'Calls':>5}  {'Total Time':>10}  "
                f"{'Avg Time':>8}  {'Tokens':>8}  {'Errors':>6}"
            )
            for name, d in sorted(dist.items(), key=lambda x: -x[1]["total_duration"]):
                avg_t = d["total_duration"] / d["count"] if d["count"] else 0
                lines.append(
                    f"  {name:<20}  {d['count']:5d}  "
                    f"{d['total_duration']:9.2f}s  {avg_t:7.2f}s  "
                    f"{d['total_tokens']:8d}  {d['errors']:6d}"
                )

        lines.append(f"\n{thin}")
        lines.append("  Token Summary")
        lines.append(thin)
        lines.append(f"  LLM Input     : {self.total_input_tokens:>10,d} tokens")
        lines.append(f"  LLM Output    : {self.total_output_tokens:>10,d} tokens")
        lines.append(
            f"  LLM Total     : {self.total_input_tokens + self.total_output_tokens:>10,d} tokens"
        )
        lines.append(f"  Retrieved     : {self.total_retrieved_tokens:>10,d} tokens")
        lines.append(f"  Total Cost    : ${self.total_cost:>9.6f}")

        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        llm_call_durations = [r.duration for r in self.llm_calls]
        tool_call_durations = [r.duration for r in self.tool_calls]
        tool_call_tokens = [float(r.retrieved_tokens) for r in self.tool_calls]
        total_tool_errors = sum(0 if r.success else 1 for r in self.tool_calls)

        return {
            "query_duration": round(self.total_duration, 3),
            "model": self.model,
            "llm_calls": [
                {
                    "call_index": i + 1,
                    "loop": r.loop,
                    "duration": round(r.duration, 3),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost": r.cost,
                    "has_tool_calls": r.has_tool_calls,
                    "forced_final": r.forced_final,
                    "phase_durations": r.phase_durations,
                }
                for i, r in enumerate(self.llm_calls)
            ],
            "tool_calls": [
                {
                    "call_index": i + 1,
                    "loop": r.loop,
                    "tool_name": r.tool_name,
                    "duration": round(r.duration, 3),
                    "retrieved_tokens": r.retrieved_tokens,
                    "success": r.success,
                    "arguments": r.arguments,
                }
                for i, r in enumerate(self.tool_calls)
            ],
            "summary": {
                "total_duration": round(self.total_duration, 3),
                "llm_duration": round(self.total_llm_duration, 3),
                "tool_duration": round(self.total_tool_duration, 3),
                "llm_call_count": len(self.llm_calls),
                "tool_call_count": len(self.tool_calls),
                "total_tool_errors": total_tool_errors,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_retrieved_tokens": self.total_retrieved_tokens,
                "total_cost": self.total_cost,
                "llm_duration_distribution": self._distribution(llm_call_durations, digits=3),
                "tool_duration_distribution": self._distribution(tool_call_durations, digits=3),
                "tool_token_distribution": self._distribution(tool_call_tokens, digits=1),
                "tool_distribution": {
                    name: {
                        "count": d["count"],
                        "total_duration": round(d["total_duration"], 3),
                        "total_tokens": d["total_tokens"],
                        "errors": d["errors"],
                    }
                    for name, d in self._tool_distribution().items()
                },
                "context_token_records": [
                    {
                        "loop": r.loop,
                        "message_tokens": r.message_tokens,
                        "system_prompt_tokens": r.system_prompt_tokens,
                        "history_tokens": r.history_tokens,
                        "latest_tool_tokens": r.latest_tool_tokens,
                    }
                    for r in self.context_token_records
                ],
            },
        }

    def log_report(self):
        report = self.format_report()
        if logger.isEnabledFor(logging.INFO):
            logger.info(report)
        else:
            print(report)


class BatchPerfAggregator:
    """批量 query 级别性能聚合器。"""

    def __init__(self):
        self.query_reports: List[Dict[str, Any]] = []

    def add_query_report(self, report: Dict[str, Any]):
        if not report or not isinstance(report, dict):
            return
        if "summary" not in report:
            return
        self.query_reports.append(report)

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = (len(sorted_values) - 1) * p
        lower = int(idx)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def _tool_distribution(self) -> Dict[str, Dict[str, Any]]:
        dist: Dict[str, Dict[str, Any]] = {}
        for report in self.query_reports:
            summary = report.get("summary", {})
            tool_dist = summary.get("tool_distribution", {})
            for name, item in tool_dist.items():
                if name not in dist:
                    dist[name] = {
                        "count": 0,
                        "total_duration": 0.0,
                        "total_tokens": 0,
                        "errors": 0,
                    }
                dist[name]["count"] += item.get("count", 0)
                dist[name]["total_duration"] += item.get("total_duration", 0.0)
                dist[name]["total_tokens"] += item.get("total_tokens", 0)
                dist[name]["errors"] += item.get("errors", 0)
        return dist

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _query_level_analysis(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for i, report in enumerate(self.query_reports, 1):
            summary = report.get("summary", {})
            llm_tokens = summary.get("total_input_tokens", 0) + summary.get(
                "total_output_tokens", 0
            )
            rows.append(
                {
                    "query_index": i,
                    "query_duration": report.get("query_duration", 0.0),
                    "llm_duration": summary.get("llm_duration", 0.0),
                    "tool_duration": summary.get("tool_duration", 0.0),
                    "llm_call_count": summary.get("llm_call_count", 0),
                    "tool_call_count": summary.get("tool_call_count", 0),
                    "total_tool_errors": summary.get("total_tool_errors", 0),
                    "llm_input_tokens": summary.get("total_input_tokens", 0),
                    "llm_output_tokens": summary.get("total_output_tokens", 0),
                    "llm_total_tokens": llm_tokens,
                    "retrieved_tokens": summary.get("total_retrieved_tokens", 0),
                    "total_cost": summary.get("total_cost", 0.0),
                }
            )
        return sorted(rows, key=lambda x: x["query_duration"], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        total_queries = len(self.query_reports)
        if total_queries == 0:
            return {
                "total_queries": 0,
                "summary": {},
            }

        query_durations = [r.get("query_duration", 0.0) for r in self.query_reports]
        llm_durations = [r.get("summary", {}).get("llm_duration", 0.0) for r in self.query_reports]
        tool_durations = [
            r.get("summary", {}).get("tool_duration", 0.0) for r in self.query_reports
        ]
        llm_total_tokens = [
            r.get("summary", {}).get("total_input_tokens", 0)
            + r.get("summary", {}).get("total_output_tokens", 0)
            for r in self.query_reports
        ]
        retrieved_tokens = [
            r.get("summary", {}).get("total_retrieved_tokens", 0) for r in self.query_reports
        ]

        total_input_tokens = sum(
            r.get("summary", {}).get("total_input_tokens", 0) for r in self.query_reports
        )
        total_output_tokens = sum(
            r.get("summary", {}).get("total_output_tokens", 0) for r in self.query_reports
        )
        total_retrieved_tokens = sum(retrieved_tokens)
        total_cost = sum(r.get("summary", {}).get("total_cost", 0.0) for r in self.query_reports)
        total_llm_calls = sum(
            r.get("summary", {}).get("llm_call_count", 0) for r in self.query_reports
        )
        total_tool_calls = sum(
            r.get("summary", {}).get("tool_call_count", 0) for r in self.query_reports
        )
        total_tool_errors = sum(
            r.get("summary", {}).get("total_tool_errors", 0) for r in self.query_reports
        )

        query_level = self._query_level_analysis()
        tool_dist = self._tool_distribution()

        tool_level_rows: List[Dict[str, Any]] = []
        for name, item in sorted(tool_dist.items(), key=lambda x: -x[1]["total_duration"]):
            calls = item.get("count", 0)
            avg_duration = item.get("total_duration", 0.0) / calls if calls else 0.0
            avg_tokens = item.get("total_tokens", 0) / calls if calls else 0.0
            error_rate = item.get("errors", 0) / calls if calls else 0.0
            tool_level_rows.append(
                {
                    "tool_name": name,
                    "call_count": calls,
                    "total_duration": round(item.get("total_duration", 0.0), 3),
                    "avg_duration": round(avg_duration, 3),
                    "total_tokens": item.get("total_tokens", 0),
                    "avg_tokens": round(avg_tokens, 2),
                    "errors": item.get("errors", 0),
                    "error_rate": round(error_rate, 4),
                    "duration_share": round(
                        item.get("total_duration", 0.0) / max(sum(tool_durations), 1e-9), 4
                    ),
                }
            )

        return {
            "total_queries": total_queries,
            "summary": {
                "total_duration": round(sum(query_durations), 3),
                "total_llm_duration": round(sum(llm_durations), 3),
                "total_tool_duration": round(sum(tool_durations), 3),
                "total_llm_calls": total_llm_calls,
                "total_tool_calls": total_tool_calls,
                "total_tool_errors": total_tool_errors,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_llm_tokens": total_input_tokens + total_output_tokens,
                "total_retrieved_tokens": total_retrieved_tokens,
                "total_cost": round(total_cost, 6),
                "avg_query_duration": round(sum(query_durations) / total_queries, 3),
                "avg_llm_call_per_query": round(total_llm_calls / total_queries, 3),
                "avg_tool_call_per_query": round(total_tool_calls / total_queries, 3),
                "avg_llm_tokens": round(
                    (total_input_tokens + total_output_tokens) / total_queries, 1
                ),
                "avg_retrieved_tokens": round(total_retrieved_tokens / total_queries, 1),
                "avg_cost_per_query": round(total_cost / total_queries, 6),
                "query_duration_distribution": {
                    "min": round(min(query_durations), 3),
                    "p50": round(self._percentile(query_durations, 0.5), 3),
                    "p90": round(self._percentile(query_durations, 0.9), 3),
                    "max": round(max(query_durations), 3),
                },
                "llm_duration_distribution": {
                    "min": round(min(llm_durations), 3),
                    "p50": round(self._percentile(llm_durations, 0.5), 3),
                    "p90": round(self._percentile(llm_durations, 0.9), 3),
                    "max": round(max(llm_durations), 3),
                },
                "tool_duration_distribution": {
                    "min": round(min(tool_durations), 3),
                    "p50": round(self._percentile(tool_durations, 0.5), 3),
                    "p90": round(self._percentile(tool_durations, 0.9), 3),
                    "max": round(max(tool_durations), 3),
                },
                "llm_token_distribution": {
                    "min": min(llm_total_tokens) if llm_total_tokens else 0,
                    "p50": round(self._percentile([float(x) for x in llm_total_tokens], 0.5), 1),
                    "p90": round(self._percentile([float(x) for x in llm_total_tokens], 0.9), 1),
                    "max": max(llm_total_tokens) if llm_total_tokens else 0,
                },
                "retrieved_token_distribution": {
                    "min": min(retrieved_tokens) if retrieved_tokens else 0,
                    "p50": round(self._percentile([float(x) for x in retrieved_tokens], 0.5), 1),
                    "p90": round(self._percentile([float(x) for x in retrieved_tokens], 0.9), 1),
                    "max": max(retrieved_tokens) if retrieved_tokens else 0,
                },
                "tool_distribution": tool_dist,
                "cost_distribution": {
                    "min": round(
                        min(
                            r.get("summary", {}).get("total_cost", 0.0) for r in self.query_reports
                        ),
                        6,
                    ),
                    "p50": round(
                        self._percentile(
                            [
                                r.get("summary", {}).get("total_cost", 0.0)
                                for r in self.query_reports
                            ],
                            0.5,
                        ),
                        6,
                    ),
                    "p90": round(
                        self._percentile(
                            [
                                r.get("summary", {}).get("total_cost", 0.0)
                                for r in self.query_reports
                            ],
                            0.9,
                        ),
                        6,
                    ),
                    "max": round(
                        max(
                            r.get("summary", {}).get("total_cost", 0.0) for r in self.query_reports
                        ),
                        6,
                    ),
                },
            },
            "query_level_analysis": query_level,
            "tool_level_analysis": tool_level_rows,
        }

    def format_report(self) -> str:
        data = self.to_dict()
        total_queries = data.get("total_queries", 0)
        if total_queries == 0:
            return "\n[ARAG Batch Performance] no query performance data."

        summary = data["summary"]
        dist = summary.get("query_duration_distribution", {})
        llm_dist = summary.get("llm_token_distribution", {})
        ret_dist = summary.get("retrieved_token_distribution", {})
        tool_dist = summary.get("tool_distribution", {})

        lines: List[str] = []
        lines.append("\n" + "=" * 72)
        lines.append("ARAG Batch Performance Summary")
        lines.append("=" * 72)
        lines.append(
            f"Queries: {total_queries} | Total time(sum): {summary.get('total_duration', 0):.2f}s | "
            f"Total cost: ${summary.get('total_cost', 0):.6f}"
        )
        lines.append(
            f"LLM calls: {summary.get('total_llm_calls', 0)} | "
            f"Tool calls: {summary.get('total_tool_calls', 0)}"
        )

        lines.append("-" * 72)
        lines.append("Timing Distribution")
        lines.append("-" * 72)
        lines.append(
            f"avg={summary.get('avg_query_duration', 0):.2f}s, "
            f"p50={dist.get('p50', 0):.2f}s, p90={dist.get('p90', 0):.2f}s, "
            f"min={dist.get('min', 0):.2f}s, max={dist.get('max', 0):.2f}s"
        )
        lines.append(
            f"LLM time(sum)={summary.get('total_llm_duration', 0):.2f}s, "
            f"Tool time(sum)={summary.get('total_tool_duration', 0):.2f}s"
        )

        lines.append("-" * 72)
        lines.append("Token Distribution")
        lines.append("-" * 72)
        lines.append(
            f"LLM input={summary.get('total_input_tokens', 0):,}, "
            f"output={summary.get('total_output_tokens', 0):,}, "
            f"total={summary.get('total_llm_tokens', 0):,}"
        )
        lines.append(
            f"Retrieved total={summary.get('total_retrieved_tokens', 0):,}, "
            f"avg/query={summary.get('avg_retrieved_tokens', 0):.1f}"
        )
        lines.append(
            f"LLM token/query: p50={llm_dist.get('p50', 0):.1f}, "
            f"p90={llm_dist.get('p90', 0):.1f}, min={llm_dist.get('min', 0)}, max={llm_dist.get('max', 0)}"
        )
        lines.append(
            f"Retrieved token/query: p50={ret_dist.get('p50', 0):.1f}, "
            f"p90={ret_dist.get('p90', 0):.1f}, min={ret_dist.get('min', 0)}, max={ret_dist.get('max', 0)}"
        )

        if tool_dist:
            lines.append("-" * 72)
            lines.append("Tool Distribution")
            lines.append("-" * 72)
            lines.append(
                f"{'Tool':<20}  {'Calls':>6}  {'Time(s)':>10}  {'Tokens':>10}  {'Errors':>6}"
            )
            for name, item in sorted(
                tool_dist.items(), key=lambda x: -x[1].get("total_duration", 0.0)
            ):
                lines.append(
                    f"{name:<20}  {item.get('count', 0):6d}  {item.get('total_duration', 0.0):10.2f}  "
                    f"{item.get('total_tokens', 0):10d}  {item.get('errors', 0):6d}"
                )

        lines.append("=" * 72)
        return "\n".join(lines)
