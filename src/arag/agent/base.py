"""Base agent implementation for ARAG."""

import json
import time
import logging
from typing import Any, Dict, List

import tiktoken

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.core.perf_tracker import QueryPerfTracker
from arag.tools.registry import ToolRegistry

logger = logging.getLogger("arag.agent")


class BaseAgent:
    """Base agent with tool calling capabilities."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        verbose: bool = False,
        perf_log: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose
        self.perf_log = perf_log
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _finalize_perf(self, tracker: QueryPerfTracker) -> Dict[str, Any]:
        """Finalize perf tracking and return structured stats."""
        tracker.finish()
        if self.perf_log:
            tracker.log_report()
        return tracker.to_dict()

    def _calculate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += len(self.tokenizer.encode(str(content)))
        return total

    def _force_final_answer(
        self,
        messages: List[Dict[str, Any]],
        context: AgentContext,
        total_cost: float,
        reason: str,
        tracker: QueryPerfTracker = None,
        loop_count: int = 0,
    ) -> tuple:
        """Force the model to give a final answer when limits are reached."""
        force_prompt = (
            "You have reached the limit. "
            "You MUST now provide a final answer based on the information you have gathered so far. "
            "Do NOT call any more tools. Synthesize the available information and respond directly."
        )

        messages.append({"role": "user", "content": force_prompt})

        try:
            forced_t0 = time.time()
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
            forced_llm_duration = time.time() - forced_t0
            total_cost += response["cost"]
            final_answer = response["message"].get("content", "")

            if tracker is not None:
                tracker.record_llm_call(
                    loop=loop_count,
                    duration=forced_llm_duration,
                    input_tokens=response.get("input_tokens", 0),
                    output_tokens=response.get("output_tokens", 0),
                    cost=response.get("cost", 0.0),
                    has_tool_calls=False,
                    forced_final=True,
                    phase_durations=response.get("phase_durations"),
                )

            if self.verbose:
                print(f"Forced answer: {final_answer[:200]}...")
                print(f"Total cost: ${total_cost:.6f}")
        except Exception as e:
            if self.verbose:
                print(f"Error getting forced answer: {e}")
            final_answer = f"Error: {reason} and failed to generate final answer."

        return final_answer, total_cost

    def run(self, query: str) -> Dict[str, Any]:
        context = AgentContext()
        tracker = QueryPerfTracker(query=query, model=self.llm.model)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        trajectory = []
        total_cost = 0.0
        loop_count = 0
        tool_schemas = self.tools.get_all_schemas()

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Question: {query}")
            print(f"{'=' * 60}\n")

        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1

            current_tokens = self._calculate_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                if self.verbose:
                    print(
                        f"Token budget exceeded ({current_tokens} > {self.max_token_budget}), forcing answer..."
                    )

                final_answer, total_cost = self._force_final_answer(
                    messages,
                    context,
                    total_cost,
                    "Token budget exceeded",
                    tracker=tracker,
                    loop_count=loop_count,
                )

                perf_data = self._finalize_perf(tracker)

                return {
                    "answer": final_answer,
                    "trajectory": trajectory,
                    "total_cost": total_cost,
                    "loops": loop_count,
                    "token_budget_exceeded": True,
                    "perf": perf_data,
                    **context.get_summary(),
                }

            if self.verbose:
                print(
                    f"Loop {loop_count}/{self.max_loops} (Tokens: {current_tokens}/{self.max_token_budget})"
                )

            system_tokens = len(self.tokenizer.encode(self.system_prompt))
            latest_tool_tokens = 0
            for msg in reversed(messages):
                if msg.get("role") in ("tool", "user") and msg.get("content"):
                    latest_tool_tokens = len(self.tokenizer.encode(str(msg["content"])))
                    break
            history_tokens = max(current_tokens - system_tokens - latest_tool_tokens, 0)
            tracker.record_context_tokens(
                loop=loop_count,
                message_tokens=current_tokens,
                system_prompt_tokens=system_tokens,
                history_tokens=history_tokens,
                latest_tool_tokens=latest_tool_tokens,
            )

            llm_t0 = time.time()
            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                if self.verbose:
                    print(f"LLM error: {e}")
                logger.exception("LLM call failed in loop %s", loop_count)
                break
            llm_duration = time.time() - llm_t0

            total_cost += response["cost"]
            message = response["message"]
            messages.append(message)

            tracker.record_llm_call(
                loop=loop_count,
                duration=llm_duration,
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                cost=response.get("cost", 0.0),
                has_tool_calls=bool(message.get("tool_calls")),
                phase_durations=response.get("phase_durations"),
            )

            if self.verbose and message.get("content"):
                print(f"Assistant: {message['content'][:200]}...")

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                final_answer = message.get("content", "")
                perf_data = self._finalize_perf(tracker)
                return {
                    "answer": final_answer,
                    "trajectory": trajectory,
                    "total_cost": total_cost,
                    "loops": loop_count,
                    "perf": perf_data,
                    **context.get_summary(),
                }

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                if self.verbose:
                    print(f"Tool: {func_name}")
                    print(f"  Args: {func_args}")

                tool_t0 = time.time()
                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                    tool_success = not bool(tool_log.get("error"))
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    tool_log = {"retrieved_tokens": 0, "error": str(e)}
                    tool_success = False
                tool_duration = time.time() - tool_t0

                tracker.record_tool_call(
                    loop=loop_count,
                    tool_name=func_name,
                    duration=tool_duration,
                    retrieved_tokens=tool_log.get("retrieved_tokens", 0),
                    success=tool_success,
                    arguments=func_args,
                )

                if self.verbose:
                    output_preview = (
                        tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                    )
                    print(f"  Result: {output_preview}")
                    if tool_log.get("retrieved_tokens", 0) > 0:
                        print(
                            f"  Tokens: {tool_log['retrieved_tokens']} | Time: {tool_duration:.2f}s"
                        )
                    print()

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )

                traj_entry = {
                    "loop": loop_count,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log,
                }
                trajectory.append(traj_entry)

        if self.verbose:
            print(f"Max loops reached ({self.max_loops}), forcing answer...")

        final_answer, total_cost = self._force_final_answer(
            messages,
            context,
            total_cost,
            "Maximum loops exceeded",
            tracker=tracker,
            loop_count=loop_count,
        )

        perf_data = self._finalize_perf(tracker)

        return {
            "answer": final_answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": loop_count,
            "max_loops_exceeded": True,
            "perf": perf_data,
            **context.get_summary(),
        }
