# Visualization Tools

A-RAG provides comprehensive visualization and analysis tools for agent execution trajectories and performance metrics.

## Features

- **Trajectory Visualization**: ASCII art rendering of agent execution steps
- **Performance Dashboard**: Detailed timing, token, and cost breakdowns
- **Batch Comparison**: Side-by-side comparison of multiple query results
- **Flexible Output**: Both text-based and structured data formats

## Quick Start

```python
from arag import LLMClient, BaseAgent, ToolRegistry
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.visualizer import visualize_result, compare_results

# Setup agent (with perf_log=True to print performance report)
agent = BaseAgent(
    llm_client=client,
    tools=tools,
    max_loops=15,
    max_token_budget=128000,
    perf_log=True,  # Enable performance tracking
)

# Run query - result includes 'perf' field
result = agent.run("Your question here?")

# Visualize single result
print(visualize_result(result))
```

## Output Format

Each `agent.run()` result now includes a `perf` field containing:

```python
{
    "answer": "...",
    "trajectory": [...],
    "total_cost": 0.005,
    "loops": 3,
    "perf": {
        "summary": {
            "total_duration": 2.45,
            "llm_duration": 1.82,
            "tool_duration": 0.63,
            "llm_call_count": 4,
            "tool_call_count": 3,
            "total_input_tokens": 12500,
            "total_output_tokens": 3200,
            "total_retrieved_tokens": 4850,
            "total_cost": 0.005,
            "llm_duration_distribution": {"mean": 0.45, "p50": 0.42, "p90": 0.55, ...},
            "tool_duration_distribution": {"mean": 0.21, "p50": 0.18, "p90": 0.32, ...},
            "context_token_records": [
                {"loop": 1, "message_tokens": 3200, "system_prompt_tokens": 800, "history_tokens": 0, "latest_tool_tokens": 0},
                {"loop": 2, "message_tokens": 5800, "system_prompt_tokens": 800, "history_tokens": 2400, "latest_tool_tokens": 1800},
                ...
            ]
        },
        "llm_calls": [...],
        "tool_calls": [...]
    }
}
```

## Visualization Examples

### Single Result Visualization

Given a result with performance data, `visualize_result()` produces:

```
======================================================================
  AGENT EXECUTION TRAJECTORY
======================================================================
─── Loop 1 ───
  [ 1] keyword_search        OK   | keywords=['institute', 'Collegian']
       Tokens: 1,213 | Result: Chunk ID: 1383, Match Score: 11.840...
  [ 2] read_chunk            OK   | chunk_ids=['58']
       Tokens: 402 | Result: [Chunk 58] Evidence Excerpt: ...

─── Loop 2 ───
  [ 3] keyword_search        OK   | keywords=['Collegian']
       Tokens: 1,850 | Result: Chunk ID: 58, Match Score: 10.838...

  TOOL USAGE STATISTICS
--------------------------------------------------
  Tool                Calls      Tokens      Time(s)
--------------------------------------------------
  keyword_search          2      3,063        0.45
  read_chunk              1        402        0.18

======================================================================

  PERFORMANCE SUMMARY
======================================================================
  Duration: 2.45s | LLM: 1.82s | Tools: 0.63s
  LLM Calls: 4 | Tool Calls: 3 | Cost: $0.005000
  Tokens - Input: 12,500 | Output: 3,200 | Retrieved: 4,850

  LLM CALL DURATIONS
----------------------------------------
  Mean: 0.455s | P50: 0.420s | P90: 0.550s

  TOOL CALL DURATIONS
----------------------------------------
  Mean: 0.210s | P50: 0.180s | P90: 0.320s

  CONTEXT TOKEN DISTRIBUTION
------------------------------------------------------------
  Loop     Total  System  History  Latest
------------------------------------------------------------
     1    3,200      800        0        0
     2    5,800      800    2,400    1,800
     3    9,200      800    5,400    2,600
```

### Batch Comparison

Compare multiple results side-by-side:

```python
from arag.visualizer import compare_results

print(compare_results(results))
```

Output:
```
======================================================================
  COMPARISON VIEW
======================================================================
  Metric                     Result 1       Result 2       Result 3
----------------------------------------------------------------------
  Duration (s)               2.45           3.21           1.98
  LLM Calls                  4              5              3
  Tool Calls                 3              4              2
  Cost ($)                   0.005000       0.006500       0.003800
  Input Tokens               12,500         15,300         9,800
  Output Tokens              3,200          4,100         2,600
  Retrieved Tokens           4,850          6,200         3,400
======================================================================
```

## Data Structures

### TrajectoryVisualizer

Renders agent execution as ASCII art.

```python
from arag.visualizer import TrajectoryVisualizer

viz = TrajectoryVisualizer(trajectory=result["trajectory"], perf_data=result["perf"])

viz.render_text()      # ASCII trajectory
viz.render_timeline()  # Timeline view with duration bars
viz.render_stats()     # Per-tool statistics
```

### PerfDashboard

Renders performance metrics.

```python
from arag.visualizer import PerfDashboard

dashboard = PerfDashboard(perf_data=result["perf"])

dashboard.render_summary()           # Overall summary
dashboard.render_timing_breakdown()  # LLM/tool timing distributions
dashboard.render_context_tokens()    # Context token per loop
```

### QueryPerfTracker

Collects per-query performance data (used internally by `BaseAgent`).

```python
from arag.core.perf_tracker import QueryPerfTracker

tracker = QueryPerfTracker(query=question, model="gpt-4o-mini")
tracker.record_llm_call(loop=1, duration=0.5, input_tokens=1000, ...)
tracker.record_tool_call(loop=1, tool_name="keyword_search", duration=0.2, ...)
tracker.finish()

perf_data = tracker.to_dict()
tracker.log_report()
```

### BatchPerfAggregator

Aggregates performance across multiple queries.

```python
from arag.core.perf_tracker import BatchPerfAggregator

aggregator = BatchPerfAggregator()
for result in results:
    aggregator.add_query_report(result["perf"])

batch_stats = aggregator.to_dict()
print(aggregator.format_report())
```

## Loading Results

Load results from JSONL files for analysis:

```python
from arag.visualizer import load_results

results = load_results("path/to/results.jsonl")
for r in results:
    print(visualize_result(r))
```

## Integration with BaseAgent

The `BaseAgent` automatically tracks performance when `perf_log=False` (default):

```python
agent = BaseAgent(
    llm_client=client,
    tools=tools,
    perf_log=False,  # Don't print report, just collect data
)

result = agent.run(query)
# result["perf"] contains structured performance data
```

Set `perf_log=True` to also print a detailed performance report after each query:

```python
agent = BaseAgent(
    llm_client=client,
    tools=tools,
    perf_log=True,  # Print report after each query
)
```

## Performance Report Example

When `perf_log=True`, the agent prints after each query:

```
====================================================================
  ARAG Performance Report
====================================================================
  Query  : When was the institute that owned The Collegian founded?
  Model  : gpt-4o-mini
  Total  : 2.45s | LLM calls: 4 | Tool calls: 3 | Cost: $0.005000

------------------------------------------------------------------------
  Timing Breakdown
--------------------------------------------------------------------
  LLM    :    1.82s ( 74.3%) | 4 calls | avg 0.45s
  Tools  :    0.63s ( 25.7%) | 3 calls | avg 0.21s
  Other  :    0.00s (  0.0%) | overhead

--------------------------------------------------------------------
  LLM Call Details
--------------------------------------------------------------------
    #    Loop  Duration    In Tok   Out Tok       Cost   Tools?  Forced?
    1       1    0.42s       2500       820  $0.000820     Yes      No
    2       1    0.48s       3200       890  $0.000980     Yes      No
    3       2    0.45s       4100       750  $0.000900     Yes      No
    4       3    0.47s       5700       740  $0.000900      No     FORCED

--------------------------------------------------------------------
  Tool Call Details
--------------------------------------------------------------------
    #    Loop  Tool                  Duration  Ret Tok  Status
    1       1  keyword_search            0.25s      1213     OK
    2       1  read_chunk               0.18s       402     OK
    3       2  keyword_search            0.20s      1850     OK

--------------------------------------------------------------------
  Tool Distribution
--------------------------------------------------------------------
  Tool                   Calls  Total Time   Avg Time    Tokens  Errors
  keyword_search             2      0.45s       0.22s       3063       0
  read_chunk                 1      0.18s       0.18s        402       0

--------------------------------------------------------------------
  Token Summary
--------------------------------------------------------------------
  LLM Input     :     12,500 tokens
  LLM Output    :      3,200 tokens
  LLM Total     :     15,700 tokens
  Retrieved     :      4,850 tokens
  Total Cost    : $0.005000

====================================================================
```

## Use Cases

### 1. Debug Agent Behavior

Analyze why certain queries fail or take too long:

```python
results = load_results("results.jsonl")
for r in results:
    if r["loops"] > 10:
        print(visualize_result(r))
```

### 2. Compare Strategies

Compare different retrieval strategies:

```python
# Compare results with different max_loops
agent_fast = BaseAgent(..., max_loops=5)
agent_deep = BaseAgent(..., max_loops=20)

# ... run queries ...

print(compare_results([result_fast, result_deep], labels=["fast", "deep"]))
```

### 3. Cost Analysis

Track API costs across queries:

```python
from arag.core.perf_tracker import BatchPerfAggregator

aggregator = BatchPerfAggregator()
for r in load_results("results.jsonl"):
    aggregator.add_query_report(r["perf"])

stats = aggregator.to_dict()
print(f"Average cost per query: ${stats['summary']['avg_cost_per_query']:.6f}")
print(f"Total cost: ${stats['summary']['total_cost']:.6f}")
```