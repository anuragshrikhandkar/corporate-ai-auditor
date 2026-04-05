"""
Corporate AI Auditor — FastAPI + Gradio
Matches the HumanAgent Interface / State Observer layout from the reference.
"""

import json
import os
from typing import Dict, Optional

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env import Action, AIAuditorEnv, TASK_REGISTRY

# ─────────────────────────────────────────────────────────────
# FastAPI backend
# ─────────────────────────────────────────────────────────────

api = FastAPI(
    title="Corporate AI Auditor — OpenEnv",
    description="AI agent environment for auditing corporate AI systems.",
    version="1.0.0",
)
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_envs: Dict[str, AIAuditorEnv] = {}


def get_env(task_id: str) -> AIAuditorEnv:
    if task_id not in _envs:
        _envs[task_id] = AIAuditorEnv(task_id=task_id)
    return _envs[task_id]


@api.get("/health")
def health():
    return {"status": "ok", "env": "ai-auditor", "tasks": list(TASK_REGISTRY.keys())}


@api.post("/reset")
def reset(task_id: str = "bias_detection"):
    return get_env(task_id).reset().model_dump()


@api.post("/step")
def step(action: Action, task_id: str = "bias_detection"):
    try:
        return get_env(task_id).step(action).model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.get("/state")
def state(task_id: str = "bias_detection"):
    return get_env(task_id).state()


@api.get("/tasks")
def list_tasks():
    return {
        tid: {"difficulty": cfg["difficulty"], "description": cfg["description"], "max_steps": cfg["max_steps"]}
        for tid, cfg in TASK_REGISTRY.items()
    }


# ─────────────────────────────────────────────────────────────
# Gradio UI — HumanAgent Interface + State Observer
# ─────────────────────────────────────────────────────────────

_gr_env: Optional[AIAuditorEnv] = None
_gr_history: list = []
_gr_obs: Optional[Dict] = None


def _fmt_obs(obs: Dict) -> str:
    sys = obs.get("ai_system", {})
    findings = obs.get("findings", [])
    docs = obs.get("documents", {})
    out = {
        "task_id": obs.get("task_id"),
        "step": obs.get("step"),
        "ai_system": {
            "name": sys.get("name"),
            "purpose": sys.get("purpose"),
            "flags": sys.get("flags", []),
        },
        "findings_so_far": len(findings),
        "documents": {k: v[:60] + "..." if len(v) > 60 else v for k, v in docs.items()},
    }
    return json.dumps(out, indent=2)


def _fmt_history() -> str:
    if not _gr_history:
        return "_No actions yet._"
    lines = []
    for h in _gr_history:
        a = h["action"]
        reward_md = f"**Reward: {h['reward']}**" if h["reward"] > 0 else f"Reward: {h['reward']}"
        lines.append(
            f"**Step {h['step']}** — `{a['action_type']}` on `{a['target']}`\n"
            f"Value: {a['value'][:80]}{'...' if len(a['value']) > 80 else ''}\n"
            f"{reward_md}\n---"
        )
    return "\n\n".join(lines)


def gr_reset(task_id):
    global _gr_env, _gr_history, _gr_obs
    _gr_env = AIAuditorEnv(task_id=task_id)
    obs = _gr_env.reset()
    _gr_obs = obs.model_dump()
    _gr_history = []
    s = _gr_env.state()
    state_md = f"**Status:** {s['status']}\n\n**Episode:** {s['episode_id'][:16]}...\n\n**Steps:** {s['step_count']}"
    return _fmt_obs(_gr_obs), "_No actions yet._", state_md, f"✅ Reset! Task: **{task_id}**"


def gr_step(action_type, target, value, reasoning):
    global _gr_env, _gr_history, _gr_obs
    if _gr_env is None:
        return "{}", "_No actions yet._", "**Status:** idle", "⚠️ Please reset first!"
    action = Action(action_type=action_type.strip(), target=target.strip(), value=value.strip(), reasoning=reasoning.strip() or None)
    try:
        result = _gr_env.step(action)
    except RuntimeError as e:
        return _fmt_obs(_gr_obs or {}), _fmt_history(), "", f"❌ {e}"

    _gr_obs = result.observation.model_dump()
    _gr_history.append({
        "step": result.observation.step,
        "action": action.model_dump(),
        "reward": result.reward,
    })
    s = _gr_env.state()
    state_md = f"**Status:** {s['status']}\n\n**Episode:** {s['episode_id'][:16]}...\n\n**Steps:** {s['step_count']}\n\n**Findings:** {s['findings_logged']}\n\n**Docs accessed:** {len(s['docs_accessed'])}"
    msg = f"✅ Step {result.observation.step} — Reward: **{result.reward}**"
    if result.done:
        fs = result.info.get("final_score", "?")
        fb = result.info.get("feedback", "")
        msg += f"\n\n🏁 **Done! Final score: {fs}** — {fb}"
    return _fmt_obs(_gr_obs), _fmt_history(), state_md, msg


def gr_get_state():
    global _gr_env, _gr_obs
    if _gr_env is None:
        return "{}", "**Status:** idle"
    s = _gr_env.state()
    state_md = f"**Status:** {s['status']}\n\n**Episode:** {(s['episode_id'] or '')[:16]}...\n\n**Steps:** {s['step_count']}\n\n**Findings:** {s['findings_logged']}\n\n**Docs accessed:** {len(s['docs_accessed'])}"
    return _fmt_obs(_gr_obs or {}), state_md


# ─────────────────────────────────────────────────────────────
# Build Gradio Blocks
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Corporate AI Auditor — OpenEnv", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Corporate AI Auditor — OpenEnv")
    gr.Markdown(
        "Train AI agents to audit corporate AI systems for bias, privacy violations, "
        "transparency failures, and regulatory breaches."
    )

    with gr.Row():

        # LEFT — HumanAgent Interface
        with gr.Column(scale=1):
            gr.Markdown("## HumanAgent Interface")

            task_dd = gr.Dropdown(
                choices=list(TASK_REGISTRY.keys()),
                value="bias_detection",
                label="Select Task",
            )

            gr.Markdown("### Take Action")

            action_type_dd = gr.Dropdown(
                choices=[
                    "request_document", "flag_bias", "flag_privacy",
                    "flag_security", "flag_transparency",
                    "assess_risk", "write_recommendation", "submit_report",
                ],
                value="request_document",
                label="Action Type *",
            )
            target_in = gr.Textbox(label="Target *", placeholder="e.g. model_card / sys_hr_001")
            value_in = gr.Textbox(label="Value *", placeholder="e.g. high: gender bias detected in feature importance", lines=3)
            reasoning_in = gr.Textbox(label="Reasoning (optional)", placeholder="Evidence from documents...")

            step_btn = gr.Button("Step", variant="primary")

            with gr.Row():
                reset_btn = gr.Button("Reset Environment", variant="secondary")
                state_btn = gr.Button("Get State", variant="secondary")

            status_box = gr.Markdown("_Reset the environment to begin._")

        # RIGHT — State Observer
        with gr.Column(scale=1):
            gr.Markdown("## State Observer")

            gr.Markdown("### Current Observation")
            obs_out = gr.Code(language="json", label="", lines=12)

            gr.Markdown("### Action History")
            hist_out = gr.Markdown("_No actions yet._")

            gr.Markdown("### Current State")
            state_out = gr.Markdown("**Status:** idle")

    gr.Markdown("**API Docs:** `/docs` &nbsp;|&nbsp; **Health:** `/health` &nbsp;|&nbsp; **Tasks:** `/tasks`")

    reset_btn.click(fn=gr_reset, inputs=[task_dd], outputs=[obs_out, hist_out, state_out, status_box])
    step_btn.click(fn=gr_step, inputs=[action_type_dd, target_in, value_in, reasoning_in], outputs=[obs_out, hist_out, state_out, status_box])
    state_btn.click(fn=gr_get_state, outputs=[obs_out, state_out])


app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
