from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .llm_runtime import MockRuntime, OpenAICompatibleRuntime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime_mode: Literal["mock", "real"] = "mock"
    llm_runtime: object | None = field(default=None)
    memory_limit: int = 3
    enable_adaptive_attempts: bool = True

    def __post_init__(self) -> None:
        if self.llm_runtime is None:
            self.llm_runtime = (
                MockRuntime()
                if self.runtime_mode == "mock"
                else OpenAICompatibleRuntime()
            )

    def _compress_memory(self, reflection_memory: list[str]) -> list[str]:
        return reflection_memory[-self.memory_limit :]

    def _should_continue(
        self,
        judge_score: int,
        judge_should_retry: bool,
        attempt_id: int,
    ) -> bool:
        if judge_score == 1:
            return False
        if attempt_id >= self.max_attempts:
            return False
        if self.agent_type != "reflexion":
            return False
        if not self.enable_adaptive_attempts:
            return True
        return judge_should_retry

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_failure_mode: str = "wrong_final_answer"

        for attempt_id in range(1, self.max_attempts + 1):
            actor_result = self.llm_runtime.actor_answer(
                example,
                attempt_id,
                self.agent_type,
                reflection_memory,
            )
            answer = actor_result.text.strip()

            judge, eval_result = self.llm_runtime.evaluate(example, answer)

            token_estimate = actor_result.total_tokens + eval_result.total_tokens
            latency_ms = actor_result.latency_ms + eval_result.latency_ms

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )

            final_answer = answer
            final_score = judge.score
            final_failure_mode = judge.error_type

            if self._should_continue(judge.score, judge.should_retry, attempt_id):
                reflection, reflect_result = self.llm_runtime.reflect(
                    example,
                    attempt_id,
                    answer,
                    judge,
                )
                trace.reflection = reflection
                trace.token_estimate += reflect_result.total_tokens
                trace.latency_ms += reflect_result.latency_ms

                reflections.append(reflection)
                reflection_memory.append(reflection.memory_update or reflection.next_strategy)
                reflection_memory = self._compress_memory(reflection_memory)

            traces.append(trace)

            if judge.score == 1 or not self._should_continue(judge.score, judge.should_retry, attempt_id):
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else final_failure_mode

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, runtime_mode: Literal["mock", "real"] = "mock") -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime_mode=runtime_mode)


class ReflexionAgent(BaseAgent):
    def __init__(
        self,
        max_attempts: int = 3,
        runtime_mode: Literal["mock", "real"] = "mock",
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            runtime_mode=runtime_mode,
        )