"""ARAG - Agentic Retrieval-Augmented Generation Framework."""

__version__ = "0.1.0"

from arag.core.config import Config
from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.core.perf_tracker import QueryPerfTracker
from arag.core.perf_tracker import BatchPerfAggregator
from arag.agent.base import BaseAgent
from arag.tools.base import BaseTool
from arag.tools.registry import ToolRegistry
from arag.visualizer import (
    TrajectoryVisualizer,
    PerfDashboard,
    visualize_result,
    compare_results,
    load_results,
)

__all__ = [
    "Config",
    "AgentContext",
    "LLMClient",
    "QueryPerfTracker",
    "BatchPerfAggregator",
    "BaseAgent",
    "BaseTool",
    "ToolRegistry",
    "TrajectoryVisualizer",
    "PerfDashboard",
    "visualize_result",
    "compare_results",
    "load_results",
    "__version__",
]
