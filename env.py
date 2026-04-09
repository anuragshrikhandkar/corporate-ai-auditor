"""
Corporate AI Auditor — env.py (ULTRA SAFE)
All scores and rewards strictly between 0 and 1, exactly in [0.01, 0.99]
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP

# Exact range from openenv.yaml
MIN_SCORE = Decimal('0.01')
MAX_SCORE = Decimal('0.99')

def safe_score(value: float) -> float:
    """Convert to Decimal, clamp to [MIN_SCORE, MAX_SCORE], never 0.0 or 1.0"""
    d = Decimal(str(value))
    if d <= 0:
        d = MIN_SCORE
    elif d >= 1:
        d = MAX_SCORE
    elif d < MIN_SCORE:
        d = MIN_SCORE
    elif d > MAX_SCORE:
        d = MAX_SCORE
    # Round to 6 decimal places to avoid floating point noise
    d = d.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
    # Final safety
    if d <= 0:
        d = MIN_SCORE
    if d >= 1:
        d = MAX_SCORE
    return float(d)

# ============ Same AISystem, AuditFinding, Observation, Action, StepResult ============
# (keep exactly as before – no changes needed)

# ============ SYSTEM DEFINITIONS ============
SYSTEM_TASK1 = AISystem(
    system_id="sys_hr_001",
    name="TalentScreen Pro",
    purpose="Automated resume screening",
    vendor="HireAI Corp",
    deployment="HR department",
    data_sources=["Historical hiring data", "Resume text"],
    model_type="XGBoost",
    last_audit="2022-01-15",
    flags=["gender_bias_risk", "proxy_discrimination"],
)

SYSTEM_TASK2 = AISystem(
    system_id="sys_health_002",
    name="PatientPredict",
    purpose="Predict patient readmission",
    vendor="MedML Solutions",
    deployment="Hospitals",
    data_sources=["EHR records", "Lab results"],
    model_type="Neural Network",
    last_audit="2021-06-01",
    flags=["gdpr_violation", "data_sharing_risk"],
)

SYSTEM_TASK3 = AISystem(
    system_id="sys_credit_003",
    name="CreditOracle v2",
    purpose="Credit scoring",
    vendor="FinScore AI",
    deployment="National bank",
    data_sources=["Credit bureau", "Transaction history"],
    model_type="Deep Neural Network",
    last_audit="2020-03-10",
    flags=["disparate_impact", "black_box_model"],
)

# ============ DOCUMENTS ============
DOCS_TASK1 = {
    "model_card": "Disparate impact ratio: 0.71. No fairness constraints.",
    "feature_importance": "Zip code proxy detected. Gender inference score: 0.09.",
    "training_data": "Historical male hire: 73%, Female: 52%",
}

DOCS_TASK2 = {
    "data_inventory": "Unencrypted S3 storage. Indefinite retention.",
    "consent_forms": "No AI consent. Last updated 2019.",
    "sharing_log": "Data shared with 3rd parties.",
    "security_report": "Critical: unencrypted data, no audit logs.",
}

DOCS_TASK3 = {
    "model_arch": "Black box, no explainability.",
    "impact_analysis": "Disparate impact ratio: 0.56. White 68%, Black 38%",
    "data_sources": "Social media data without consent.",
    "regulatory": "CFPB inquiry, ECOA complaint.",
    "financial": "Potential fines: $500M-$2B",
}

# ============ REQUIRED FLAGS ============
TASK1_REQUIRED_FLAGS = {
    "gender_bias": {"action": "flag_bias", "keywords": ["gender", "female", "male", "0.71"]},
    "proxy_discrimination": {"action": "flag_bias", "keywords": ["zip", "proxy", "prestige"]},
    "historical_bias": {"action": "flag_bias", "keywords": ["historical", "73", "52"]},
}

TASK2_REQUIRED_FLAGS = {
    "consent": {"action": "flag_privacy", "keywords": ["consent", "2019"]},
    "data_sharing": {"action": "flag_privacy", "keywords": ["sharing", "third"]},
    "security": {"action": "flag_security", "keywords": ["unencrypted", "s3"]},
    "retention": {"action": "flag_privacy", "keywords": ["retention", "indefinite"]},
}

TASK3_REQUIRED_FLAGS = {
    "disparate_impact": {"action": "flag_bias", "keywords": ["disparate", "0.56", "38"]},
    "explainability": {"action": "flag_transparency", "keywords": ["black box", "explain"]},
    "privacy_breach": {"action": "flag_privacy", "keywords": ["social media", "consent"]},
    "regulatory": {"action": "flag_bias", "keywords": ["cfpb", "ecoa"]},
    "risk": {"action": "assess_risk", "keywords": ["500m", "2b"]},
    "recommendation": {"action": "write_recommendation", "keywords": ["remediate", "suspend"]},
}

TASK_REGISTRY = {
    "bias_detection": {
        "id": "bias_detection",
        "description": "Audit TalentScreen Pro for bias",
        "ai_system": SYSTEM_TASK1,
        "documents": DOCS_TASK1,
        "required_flags": TASK1_REQUIRED_FLAGS,
        "max_steps": 12,
    },
    "privacy_compliance": {
        "id": "privacy_compliance",
        "description": "Audit PatientPredict for privacy",
        "ai_system": SYSTEM_TASK2,
        "documents": DOCS_TASK2,
        "required_flags": TASK2_REQUIRED_FLAGS,
        "max_steps": 16,
    },
    "full_risk_audit": {
        "id": "full_risk_audit",
        "description": "Full audit of CreditOracle",
        "ai_system": SYSTEM_TASK3,
        "documents": DOCS_TASK3,
        "required_flags": TASK3_REQUIRED_FLAGS,
        "max_steps": 25,
    },
}

# ============ GRADING FUNCTIONS ============
def _match_finding(action, spec):
    if action.action_type != spec["action"]:
        return 0.01
    text = (action.value + " " + (action.reasoning or "")).lower()
    matches = sum(1 for kw in spec["keywords"] if kw.lower() in text)
    if matches == 0:
        return 0.01
    proportion = matches / len(spec["keywords"])
    # Scale from 0.01 to 0.99
    score = 0.01 + (proportion * 0.98)
    return safe_score(score)

def _best(actions, spec):
    if not actions:
        return 0.01
    scores = [_match_finding(a, spec) for a in actions]
    return safe_score(max(scores))

def grade_task1(actions):
    scores = {k: _best(actions, spec) for k, spec in TASK1_REQUIRED_FLAGS.items()}
    total = sum(scores.values()) / len(TASK1_REQUIRED_FLAGS)
    return safe_score(total), scores, f"{sum(1 for v in scores.values() if v>0.5)}/{len(TASK1_REQUIRED_FLAGS)} issues found"

def grade_task2(actions):
    scores = {k: _best(actions, spec) for k, spec in TASK2_REQUIRED_FLAGS.items()}
    total = sum(scores.values()) / len(TASK2_REQUIRED_FLAGS)
    return safe_score(total), scores, f"{sum(1 for v in scores.values() if v>0.5)}/{len(TASK2_REQUIRED_FLAGS)} issues found"

def grade_task3(actions):
    scores = {k: _best(actions, spec) for k, spec in TASK3_REQUIRED_FLAGS.items()}
    total = sum(scores.values()) / len(TASK3_REQUIRED_FLAGS)
    return safe_score(total), scores, f"{sum(1 for v in scores.values() if v>0.5)}/{len(TASK3_REQUIRED_FLAGS)} issues found"

GRADERS = {
    "bias_detection": grade_task1,
    "privacy_compliance": grade_task2,
    "full_risk_audit": grade_task3,
}

# ============ ENVIRONMENT ============
class AIAuditorEnv:
    def __init__(self, task_id: str = "bias_detection"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_id}")
        self.task_id = task_id
        self._cfg = TASK_REGISTRY[task_id]
        self._episode_id = None
        self._step_count = 0
        self._done = False
        self._actions_taken = []
        self._action_history = []
        self._findings = []
        self._docs_accessed = []
        self._current_obs = None
        self._status = "idle"

    def reset(self) -> Observation:
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._actions_taken = []
        self._action_history = []
        self._findings = []
        self._docs_accessed = []
        self._status = "running"
        self._current_obs = self._build_obs()
        return self._current_obs

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done")
        if self._current_obs is None:
            raise RuntimeError("Call reset first")
        
        self._step_count += 1
        self._actions_taken.append(action)
        self._apply_action(action)
        
        self._action_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": 0.0,
        })
        
        submitted = action.action_type == "submit_report"
        self._done = self._step_count >= self._cfg["max_steps"] or submitted
        
        if self._done:
            self._status = "done"
            final_score, breakdown, feedback = GRADERS[self.task_id](self._actions_taken)
            final_score = safe_score(final_score)
            info = {
                "final_score": final_score,
                "breakdown": breakdown,
                "feedback": feedback,
                "episode_id": self._episode_id,
                "steps": self._step_count,
            }
            # Reward when done: use final_score directly, scaled but never <0.01
            reward = final_score
        else:
            info = {"episode_id": self._episode_id, "steps": self._step_count}
            current_score, _, _ = GRADERS[self.task_id](self._actions_taken)
            # Reward based on progress, but guarantee min 0.01
            reward = max(0.01, min(0.99, current_score * (self._step_count / self._cfg["max_steps"])))
        
        reward = safe_score(reward)
        if self._action_history:
            self._action_history[-1]["reward"] = reward
        
        self._current_obs = self._build_obs()
        
        return StepResult(
            observation=self._current_obs,
            reward=reward,
            done=self._done,
            info=info
        )

    def state(self) -> Dict[str, Any]:
        return {
            "status": self._status,
            "episode_id": self._episode_id,
            "task_id": self.task_id,
            "step_count": self._step_count,
            "done": self._done,
            "findings_logged": len(self._findings),
            "docs_accessed": self._docs_accessed,
            "action_history": self._action_history,
        }

    def _build_obs(self) -> Observation:
        doc_index = {}
        for k, v in self._cfg["documents"].items():
            if k in self._docs_accessed:
                doc_index[k] = f"[ACCESSED] {v}"
            else:
                doc_index[k] = "[NOT YET ACCESSED]"
        return Observation(
            task_id=self.task_id,
            task_description=self._cfg["description"],
            ai_system=self._cfg["ai_system"],
            findings=self._findings.copy(),
            documents=doc_index,
            step=self._step_count,
            max_steps=self._cfg["max_steps"],
            context={
                "episode_id": self._episode_id,
                "docs_accessed": self._docs_accessed,
            },
        )

    def _apply_action(self, action: Action):
        if action.action_type == "request_document":
            if action.target in self._cfg["documents"] and action.target not in self._docs_accessed:
                self._docs_accessed.append(action.target)
        elif action.action_type in ["flag_bias", "flag_privacy", "flag_security", 
                                     "flag_transparency", "assess_risk", "write_recommendation"]:
            self._findings.append(AuditFinding(
                finding_id=f"F{len(self._findings)+1:03d}",
                category=action.action_type,
                severity="high",
                description=action.value[:200],
                evidence=action.reasoning or "",
                status="open",
            ))
