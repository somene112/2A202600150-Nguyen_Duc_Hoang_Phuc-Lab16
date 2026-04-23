# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.85 | 0.86 | 0.01 |
| Avg attempts | 1 | 1.06 | 0.06 |
| Avg token estimate | 3307.61 | 3557.93 | 250.32 |
| Avg latency (ms) | 3492.97 | 3881.21 | 388.24 |

## Failure modes
```json
{
  "overall": {
    "none": 171,
    "wrong_final_answer": 26,
    "entity_drift": 2,
    "incomplete_multi_hop": 1
  },
  "react": {
    "none": 85,
    "wrong_final_answer": 13,
    "entity_drift": 1,
    "incomplete_multi_hop": 1
  },
  "reflexion": {
    "none": 86,
    "wrong_final_answer": 13,
    "entity_drift": 1
  }
}
Extensions implemented

- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding
- adaptive_max_attempts
- memory_compression

Discussion

This benchmark compares a single-attempt ReAct baseline against a multi-attempt Reflexion agent in real mode. The Reflexion system is expected to improve exact match when the first attempt fails because of incomplete multi-hop reasoning or entity drift. In this run, ReAct achieved EM=0.85 while Reflexion achieved EM=0.86, for an absolute delta of 0.01. That gain comes with higher average attempts, token usage, and latency because Reflexion pays for evaluator and reflector calls in failed trajectories. The most common failure mode across all records was 'none', which is a useful diagnostic because it reveals whether the actor is failing on first-hop retrieval, second-hop grounding, or final answer selection. If Reflexion reduces incomplete_multi_hop more than wrong_final_answer, that usually means the reflection memory is helping the model remember to finish the second reasoning hop but not yet forcing enough evidence checking at the end. A strong final report should also inspect example traces to see whether the evaluator feedback was precise enough to guide the next attempt. If the evaluator is vague, Reflexion can spend extra tokens without meaningful gains; if the evaluator is structured and specific, reflection memory becomes much more valuable. The benchmark format used here is intentionally compatible with autograding, but the real quality signal comes from whether later attempts visibly change strategy instead of merely restating the same wrong answer.
