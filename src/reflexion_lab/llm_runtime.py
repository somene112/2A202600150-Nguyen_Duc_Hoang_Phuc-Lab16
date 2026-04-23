from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from urllib import request, error

from dotenv import load_dotenv

from .mock_runtime import FAILURE_MODE_BY_QID, FIRST_ATTEMPT_WRONG
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()


def _format_context(example: QAExample) -> str:
    return "\n\n".join(
        f"[{idx}] {chunk.title}\n{chunk.text}"
        for idx, chunk in enumerate(example.context, start=1)
    )


def _estimate_tokens_from_text(*parts: str) -> int:
    text = " ".join(part for part in parts if part)
    return max(1, len(text) // 4)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("No JSON object found in model output")


@dataclass
class LLMCallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OpenAICompatibleRuntime:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = (
            float(os.getenv("TEMPERATURE", "0"))
            if temperature is None
            else float(temperature)
        )

        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 220,
    ) -> LLMCallResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        req_data = json.dumps(payload).encode("utf-8")

        max_retries = 6
        base_wait = 3.0

        for attempt in range(max_retries):
            req = request.Request(
                url=f"{self.base_url}/chat/completions",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )

            started = time.perf_counter()
            try:
                with request.urlopen(req, timeout=180) as response:
                    body = response.read().decode("utf-8")
                latency_ms = int((time.perf_counter() - started) * 1000)

                data = json.loads(body)
                choice = data["choices"][0]["message"]
                text = choice.get("content", "") or ""

                usage = data.get("usage", {})
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)

                if prompt_tokens == 0 and completion_tokens == 0:
                    prompt_tokens = _estimate_tokens_from_text(system_prompt, user_prompt)
                    completion_tokens = _estimate_tokens_from_text(text)

                return LLMCallResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )

            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                if exc.code == 429 and attempt < max_retries - 1:
                    wait_time = base_wait * (2 ** attempt) + random.uniform(0, 1.0)
                    print(f"[WARN] OpenAI rate limit hit. Retry in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"OpenAI HTTP error {exc.code}: {detail}") from exc

            except Exception as exc:
                if attempt < max_retries - 1:
                    wait_time = base_wait * (2 ** attempt) + random.uniform(0, 1.0)
                    print(f"[WARN] Request failed: {exc}. Retry in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"Failed to call OpenAI API: {exc}") from exc

        raise RuntimeError("Exceeded maximum retries for OpenAI request.")

    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> LLMCallResult:
        reflection_block = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "None"

        user_prompt = f"""
Question: {example.question}

Context:
{_format_context(example)}

Agent type: {agent_type}
Attempt: {attempt_id}
Reflection memory:
{reflection_block}

Return only the final answer text.
""".strip()

        return self._chat(ACTOR_SYSTEM, user_prompt, max_tokens=80)

    def evaluate(self, example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallResult]:
        user_prompt = f"""
Question: {example.question}
Gold answer: {example.gold_answer}
Predicted answer: {answer}

Context:
{_format_context(example)}

Return STRICT JSON only with keys:
score, reason, missing_evidence, spurious_claims, confidence, error_type, should_retry
""".strip()

        result = self._chat(EVALUATOR_SYSTEM, user_prompt, max_tokens=220)

        try:
            payload = _extract_json_object(result.text)
            judge = JudgeResult.model_validate(payload)
        except Exception:
            judge = self._heuristic_evaluator(example, answer)

        return judge, result

    def evaluate_local(self, example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallResult]:
        judge = self._heuristic_evaluator(example, answer)
        result = LLMCallResult(
            text=judge.model_dump_json(),
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=1,
        )
        return judge, result

    def reflect(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, LLMCallResult]:
        user_prompt = f"""
Question: {example.question}
Attempt: {attempt_id}
Previous answer: {answer}
Evaluator reason: {judge.reason}
Missing evidence: {judge.missing_evidence}
Spurious claims: {judge.spurious_claims}

Return STRICT JSON only with keys:
failure_reason, lesson, next_strategy, memory_update
""".strip()

        result = self._chat(REFLECTOR_SYSTEM, user_prompt, max_tokens=180)

        try:
            payload = _extract_json_object(result.text)
            reflection = ReflectionEntry.model_validate(
                {
                    "attempt_id": attempt_id,
                    "failure_reason": payload.get("failure_reason", judge.reason),
                    "lesson": payload.get("lesson", "Complete all reasoning hops before answering."),
                    "next_strategy": payload.get(
                        "next_strategy",
                        "Re-read the second supporting paragraph before finalizing."
                    ),
                    "memory_update": payload.get(
                        "memory_update",
                        payload.get("lesson", "")
                    ),
                }
            )
        except Exception:
            reflection = self._heuristic_reflector(example, attempt_id, judge)

        return reflection, result

    def _heuristic_evaluator(self, example: QAExample, answer: str) -> JudgeResult:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(
                score=1,
                reason="Final answer matches the gold answer after normalization.",
                confidence=0.99,
                error_type="none",
                should_retry=False,
            )

        if normalize_answer(answer) == "london":
            return JudgeResult(
                score=0,
                reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
                missing_evidence=["Need to identify the river that flows through London."],
                spurious_claims=[],
                confidence=0.86,
                error_type="incomplete_multi_hop",
                should_retry=True,
            )

        return JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
            confidence=0.72,
            error_type=FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer"),
            should_retry=True,
        )

    def _heuristic_reflector(
        self,
        example: QAExample,
        attempt_id: int,
        judge: JudgeResult,
    ) -> ReflectionEntry:
        strategy = (
            "Do the second hop explicitly: birthplace city -> river through that city."
            if example.qid == "hp2"
            else "Verify the final entity against the second paragraph before answering."
        )

        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
            next_strategy=strategy,
            memory_update=strategy,
        )


class MockRuntime:
    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> LLMCallResult:
        if example.qid not in FIRST_ATTEMPT_WRONG:
            answer = example.gold_answer
        elif agent_type == "react":
            answer = FIRST_ATTEMPT_WRONG[example.qid]
        elif attempt_id == 1 and not reflection_memory:
            answer = FIRST_ATTEMPT_WRONG[example.qid]
        else:
            answer = example.gold_answer

        prompt_tokens = _estimate_tokens_from_text(
            example.question,
            _format_context(example),
            " ".join(reflection_memory),
        )
        completion_tokens = _estimate_tokens_from_text(answer)

        return LLMCallResult(
            text=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=40 + (attempt_id * 10) + (15 if agent_type == "reflexion" else 0),
        )

    def evaluate(self, example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallResult]:
        judge = self._heuristic_evaluator(example, answer)
        result = LLMCallResult(
            text=judge.model_dump_json(),
            prompt_tokens=_estimate_tokens_from_text(example.question, answer),
            completion_tokens=_estimate_tokens_from_text(judge.reason),
            latency_ms=25,
        )
        return judge, result

    def evaluate_local(self, example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallResult]:
        return self.evaluate(example, answer)

    def reflect(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, LLMCallResult]:
        reflection = self._heuristic_reflector(example, attempt_id, judge)
        result = LLMCallResult(
            text=reflection.model_dump_json(),
            prompt_tokens=_estimate_tokens_from_text(example.question, answer, judge.reason),
            completion_tokens=_estimate_tokens_from_text(reflection.lesson, reflection.next_strategy),
            latency_ms=20,
        )
        return reflection, result

    def _heuristic_evaluator(self, example: QAExample, answer: str) -> JudgeResult:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(
                score=1,
                reason="Final answer matches the gold answer after normalization.",
                confidence=0.99,
                error_type="none",
                should_retry=False,
            )
        if normalize_answer(answer) == "london":
            return JudgeResult(
                score=0,
                reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
                missing_evidence=["Need to identify the river that flows through London."],
                spurious_claims=[],
                confidence=0.90,
                error_type="incomplete_multi_hop",
                should_retry=True,
            )
        return JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
            confidence=0.80,
            error_type=FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer"),
            should_retry=True,
        )

    def _heuristic_reflector(
        self,
        example: QAExample,
        attempt_id: int,
        judge: JudgeResult,
    ) -> ReflectionEntry:
        strategy = (
            "Do the second hop explicitly: birthplace city -> river through that city."
            if example.qid == "hp2"
            else "Verify the final entity against the second paragraph before answering."
        )
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
            next_strategy=strategy,
            memory_update=strategy,
        )