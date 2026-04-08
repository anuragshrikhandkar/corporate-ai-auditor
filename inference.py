"""
inference.py — Corporate AI Auditor Baseline (Updated)
"""

import json
import os
from openai import OpenAI
from env import Action, AIAuditorEnv, TASK_REGISTRY

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")

MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 400
SUCCESS_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# SYSTEM_PROMPT same as before (copy your previous one)

def build_prompt(obs: dict, history: list) -> str:
    # (your previous build_prompt function - same as you sent)
    ...  # paste your full build_prompt here

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
            unread = [k for k, v in docs.items() if "NOT ACCESSED" in v]
            if unread:
                action = Action(action_type="request_document", target=unread[0], value="read")
                action_str = f"request_document('{unread[0]}')[fallback]"
            else:
                action = Action(action_type="flag_bias", target=obs_dict["ai_system"]["system_id"], value="potential bias detected")
                action_str = "flag_bias(fallback)"

        result = env.step(action)
        step += 1
        obs = result.observation
        done = result.done
        reward = result.reward
        rewards.append(reward)
        history.append({
            "step": step,
            "action_type": action.action_type,
            "target": action.target,
            "reward": reward,
        })

        error_str = last_error if last_error else "null"
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
            flush=True,
        )

    # === STRONG FINAL CLAMP ===
    final_score = 0.05
    if result and result.info.get("final_score") is not None:
        fs = float(result.info["final_score"])
        final_score = max(0.01, min(0.99, fs))
    elif rewards:
        raw_fs = sum(rewards) / max(len(rewards), 1)
        final_score = max(0.01, min(0.99, raw_fs))

    final_score = round(final_score, 4)   # extra safety

    success = final_score >= SUCCESS_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)
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
        print(f"[{status}] {r['task_id']:<22} score={r['final_score']:.4f} steps={r['steps']}")
