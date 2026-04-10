"""
inference.py — Corporate AI Auditor Baseline (FINAL FIX v2)
"""

import json
import os
from openai import OpenAI
from env import Action, AIAuditorEnv, TASK_REGISTRY

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")

MAX_STEPS         = 20
TEMPERATURE       = 0.2
MAX_TOKENS        = 400
SUCCESS_THRESHOLD = 0.5

_LO = 0.01
_HI = 0.99

def _clamp(v):
    try:
        v = float(v)
    except Exception:
        return _LO
    if v != v:
        return _LO
    if v <= 0.0 or v < _LO:
        return _LO
    if v >= 1.0 or v > _HI:
        return _HI
    return round(v, 4)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert AI ethics auditor. You audit corporate AI systems for:
- Bias and discrimination (race, gender, age, proxy features)
- Privacy violations (GDPR, HIPAA, consent, data minimization)
- Security gaps (encryption, access controls, penetration testing)
- Transparency failures (explainability, reason codes, black-box models)
- Regulatory breaches (ECOA, CFPB, GDPR, CCPA)
You are methodical: first request and read documents, then flag issues with evidence.
Always cite specific numbers, percentages, or facts from the documents in your findings."""

def build_prompt(obs: dict, history: list) -> str:
    sys = obs["ai_system"]
    docs = obs["documents"]
    findings = obs["findings"]

    docs_summary = "\n".join(
        f"  [{k}]: {v[:150]}..." if len(v) > 150 else f"  [{k}]: {v}"
        for k, v in docs.items()
    )
    findings_summary = (
        "\n".join(f"  - {f['category']}: {f['description'][:80]}" for f in findings)
        if findings else "  None yet."
    )
    history_summary = ""
    if history:
        history_summary = "\nRecent actions:\n" + "\n".join(
            f"  step={h['step']} {h['action_type']}({h['target']}) reward={h['reward']:.4f}"
            for h in history[-5:]
        )

    return f"""TASK: {obs['task_description']}

AI SYSTEM UNDER AUDIT:
  Name: {sys['name']}
  Purpose: {sys['purpose']}
  Deployment: {sys['deployment']}
  Data sources: {', '.join(sys['data_sources'])}
  Risk flags: {', '.join(sys['flags'])}

DOCUMENTS (use request_document to access unread ones):
{docs_summary}

FINDINGS LOGGED SO FAR:
{findings_summary}

STEP: {obs['step']}/{obs['max_steps']}
{history_summary}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "action_type": "<request_document|flag_bias|flag_privacy|flag_security|flag_transparency|assess_risk|write_recommendation|submit_report>",
  "target": "<document_name or system_id or finding_id>",
  "value": "<specific finding with evidence, numbers, regulation names>",
  "reasoning": "<cite exact evidence from documents>"
}}

Priority: request unread documents first, then flag specific issues with evidence."""

def run_task(task_id: str) -> dict:
    env = AIAuditorEnv(task_id=task_id)
    obs = env.reset()
    print(f"[START] task={task_id} env=ai-auditor model={MODEL_NAME}", flush=True)

    rewards = []
    history = []
    done = False
    step = 0
    result = None
    last_error = None

    while not done and step < MAX_STEPS:
        obs_dict = obs.model_dump()
        prompt = build_prompt(obs_dict, history)
        action_str = "null"
        last_error = None

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            parsed = json.loads(raw)
            action = Action(**parsed)
            action_str = f"{action.action_type}('{action.target}')"
        except Exception as e:
            last_error = str(e)[:80]
            docs = obs_dict.get("documents", {})
            unread = [k for k, v in docs.items() if "NOT YET" in v]
            if unread:
                action = Action(action_type="request_document", target=unread[0], value="read")
                action_str = f"request_document('{unread[0]}')[fallback]"
            else:
                action = Action(action_type="flag_bias", target=obs_dict["ai_system"]["system_id"],
                                value="potential bias detected in historical training data")
                action_str = "flag_bias(fallback)"

        result = env.step(action)
        step += 1
        obs = result.observation
        done = result.done
        reward = _clamp(result.reward)
        rewards.append(reward)

        history.append({
            "step": step,
            "action_type": action.action_type,
            "target": action.target,
            "reward": reward,
        })

        error_str = last_error if last_error else "null"
        print(f"[STEP] step={step} action={action_str} reward={reward:.4f} done={str(done).lower()} error={error_str}", flush=True)

    if result and result.info.get("final_score") is not None:
        final_score = _clamp(result.info["final_score"])
    elif rewards:
        final_score = _clamp(sum(rewards) / len(rewards))
    else:
        final_score = _LO

    success = final_score >= SUCCESS_THRESHOLD
    if not rewards:
        rewards = [_LO]
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} final_score={final_score:.4f} rewards={rewards_str}", flush=True)
    return {"task_id": task_id, "success": success, "final_score": final_score, "steps": step}

if __name__ == "__main__":
    print("=" * 60)
    print("Corporate AI Auditor — Baseline Inference")
    print("=" * 60)
    print()
    results = []
    for task_id in TASK_REGISTRY.keys():
        r = run_task(task_id)
        results.append(r)
        print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"[{status}] {r['task_id']:<22} score={r['final_score']:.4f}  steps={r['steps']}")
