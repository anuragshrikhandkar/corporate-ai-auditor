"""
Corporate AI Auditor — env.py (FINAL SHIELD v6)
Strictly (0.02, 0.98) - Absolute Zero Elimination
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# ✅ STRICT BOUNDARIES - NO ZEROS ALLOWED
_LO = 0.02
_HI = 0.98

def _clamp(v: float) -> float:
    try:
        v = float(v)
    except:
        return _LO
    if v != v: return _LO # NaN check
    # Strictly force away from 0.0 and 1.0
    return round(max(_LO, min(_HI, v)), 4)

class AISystem(BaseModel):
    system_id: str
    name: str
    purpose: str
    vendor: str
    deployment: str
    data_sources: List[str]
    model_type: str
    last_audit: Optional[str] = None
    flags: List[str] = []

class AuditFinding(BaseModel):
    finding_id: str
    category: str
    severity: str
    description: str
    evidence: str
    status: str = "open"

class Observation(BaseModel):
    task_id: str
    task_description: str
    ai_system: AISystem
    findings: List[AuditFinding]
    documents: Dict[str, str]
    step: int
    max_steps: int
    context: Dict[str, Any] = {}

class Action(BaseModel):
    action_type: str = Field(description="Action categories")
    target: str = Field(description="Target ID")
    value: str = Field(description="Finding text")
    reasoning: Optional[str] = None

class Reward(BaseModel):
    score: float 
    breakdown: Dict[str, float] = {}
    feedback: str = ""

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}

# ══════════════════════════════════════════════════════════════
# SYSTEMS & DATA
# ══════════════════════════════════════════════════════════════

SYSTEM_TASK1 = AISystem(
    system_id="sys_hr_001", name="TalentScreen Pro",
    purpose="Automated resume screening and candidate ranking",
    vendor="HireAI Corp", deployment="HR dept",
    data_sources=["Historical data", "LinkedIn profiles"],
    model_type="XGBoost", flags=["historical_bias_risk", "zip_code_proxy"]
)

SYSTEM_TASK2 = AISystem(
    system_id="sys_health_002", name="PatientPredict",
    purpose="Predicts patient readmission risk",
    vendor="MedML", deployment="Hospitals",
    data_sources=["EHR records", "Claims"],
    model_type="Neural Network", flags=["pii_detected", "gdpr_scope"]
)

SYSTEM_TASK3 = AISystem(
    system_id="sys_credit_003", name="CreditOracle v2",
    purpose="Automated credit scoring",
    vendor="FinScore AI", deployment="National Bank",
    data_sources=["Credit bureau", "Bank history"],
    model_type="Deep Learning", flags=["ecoa_violation", "disparate_impact"]
)

DOCS_TASK1 = {
    "model_card": "Accuracy: 89%. Disparate impact ratio (female/male): 0.71.",
    "feature_importance": "Top: University prestige (0.31), Zip code (0.12).",
    "training_data_report": "Labels from 2015-2022. Male hire rate 73%, Female 52%.",
    "vendor_contract": "Audit rights limited. Right to explanation not included."
}

DOCS_TASK2 = {
    "data_inventory": "Data: Name, SSN, diagnosis. Storage: AWS S3 unencrypted. Retention: indefinite.",
    "consent_forms": "General consent only. No specific AI sharing consent. Last updated 2019.",
    "data_sharing_log": "Shared with MedML, pharma companies. Re-id risk: high.",
    "security_report": "Critical: unencrypted S3, default credentials. 1 of 3 fixed."
}

DOCS_TASK3 = {
    "model_architecture": "Black box output 0-999. No interpretability layer.",
    "disparate_impact_analysis": "Impact ratio: 0.56 (below 0.80 legal threshold).",
    "data_sources_detail": "GPS tracking via mobile app (undisclosed). Data purchased from brokers.",
    "regulatory_correspondence": "CFPB inquiry received 2023-08-14. ECOA complaint filed.",
    "financial_impact": "Potential fine: $500M-$2B. Reputational risk: Critical."
}

# ══════════════════════════════════════════════════════════════
# FLAGS & LOGIC
# ══════════════════════════════════════════════════════════════

TASK1_REQUIRED_FLAGS = {
    "gender_bias": {"action": "flag_bias", "keywords": ["gender","female","male","0.71"]},
    "proxy_discrimination": {"action": "flag_bias", "keywords": ["zip","proxy","university"]},
    "historical_bias": {"action": "flag_bias", "keywords": ["historical","training","2015"]}
}

TASK2_REQUIRED_FLAGS = {
    "gdpr_consent": {"action": "flag_privacy", "keywords": ["consent","gdpr","2019"]},
    "data_sharing": {"action": "flag_privacy", "keywords": ["sharing","pharma","third"]},
    "security_gaps": {"action": "flag_security", "keywords": ["encrypt","s3","critical"]},
    "retention": {"action": "flag_privacy", "keywords": ["retention","indefinite"]}
}

TASK3_REQUIRED_FLAGS = {
    "disparate_impact": {"action": "flag_bias", "keywords": ["disparate","0.56","0.80","ecoa"]},
    "explainability": {"action": "flag_transparency", "keywords": ["black box","interpret"]},
    "unlawful_data": {"action": "flag_privacy", "keywords": ["gps","location","broker"]},
    "regulatory_breach": {"action": "flag_bias", "keywords": ["cfpb","ecoa","legal"]},
    "risk_assessment": {"action": "assess_risk", "keywords": ["critical","2b","fine"]},
    "written_report": {"action": "write_recommendation", "keywords": ["halt","suspend","immediate"]}
}

TASK_REGISTRY = {
    "bias_detection": {"id": "bias_detection", "ai_system": SYSTEM_TASK1, "documents": DOCS_TASK1, "required_flags": TASK1_REQUIRED_FLAGS, "max_steps": 12, "description": "Audit for bias."},
    "privacy_compliance": {"id": "privacy_compliance", "ai_system": SYSTEM_TASK2, "documents": DOCS_TASK2, "required_flags": TASK2_REQUIRED_FLAGS, "max_steps": 16, "description": "Audit for privacy."},
    "full_risk_audit": {"id": "full_risk_audit", "ai_system": SYSTEM_TASK3, "documents": DOCS_TASK3, "required_flags": TASK3_REQUIRED_FLAGS, "max_steps": 25, "description": "Full risk audit."}
}

def _match_finding(action, spec) -> float:
    if action.action_type != spec["action"]:
        return _LO
    text = (action.value + " " + (action.reasoning or "")).lower()
    hits = sum(1 for kw in spec["keywords"] if kw.lower() in text)
    res = hits / max(2, len(spec["keywords"]) // 2)
    return _clamp(res)

def _best(actions, spec) -> float:
    return max((_match_finding(a, spec) for a in actions), default=_LO)

def grade_task(actions, required):
    # Strictly clamp every individual score in the breakdown
    scores = {k: _clamp(_best(actions, spec)) for k, spec in required.items()}
    total = sum(scores.values()) / len(required)
    final = _clamp(total)
    return final, scores, f"Audit complete. Score: {final}"

GRADERS = {
    "bias_detection": lambda acts: grade_task(acts, TASK1_REQUIRED_FLAGS),
    "privacy_compliance": lambda acts: grade_task(acts, TASK2_REQUIRED_FLAGS),
    "full_risk_audit": lambda acts: grade_task(acts, TASK3_REQUIRED_FLAGS),
}

# ══════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class AIAuditorEnv:
    def __init__(self, task_id: str = "bias_detection"):
        self.task_id = task_id
        self._cfg = TASK_REGISTRY[task_id]
        self._step_count = 0
        self._actions_taken = []
        self._docs_accessed = []
        self._findings = []
        self._done = False

    def reset(self) -> Observation:
        self._step_count = 0
        self._done = False
        self._actions_taken = []
        self._docs_accessed = []
        self._findings = []
        return self._build_obs()

    def step(self, action: Action) -> StepResult:
        if self._done: raise RuntimeError("Episode done.")
        self._step_count += 1
        self._actions_taken.append(action)
        self._apply_action(action)
        
        safe_reward = _clamp(self._compute_step_reward(action))
        
        self._done = self._step_count >= self._cfg["max_steps"] or action.action_type == "submit_report"
        
        info = {"steps": self._step_count}
        if self._done:
            f_score, f_breakdown, f_feedback = GRADERS[self.task_id](self._actions_taken)
            info.update({
                "final_score": _clamp(f_score),
                "breakdown": {k: _clamp(v) for k, v in f_breakdown.items()},
                "feedback": f_feedback
            })
            
        return StepResult(observation=self._build_obs(), reward=safe_reward, done=self._done, info=info)

    def _compute_step_reward(self, action: Action) -> float:
        score_now, _, _ = GRADERS[self.task_id](self._actions_taken)
        prev = self._actions_taken[:-1]
        score_prev = GRADERS[self.task_id](prev)[0] if prev else _LO
        delta = score_now - score_prev
        bonus = 0.05 if (action.action_type == "request_document" and action.target in self._cfg["documents"] and action.target not in self._docs_accessed[:-1]) else 0.0
        return delta + bonus + 0.01

    def _apply_action(self, action: Action):
        if action.action_type == "request_document" and action.target in self._cfg["documents"]:
            if action.target not in self._docs_accessed: self._docs_accessed.append(action.target)
        elif "flag_" in action.action_type or action.action_type in ["assess_risk", "write_recommendation"]:
            self._findings.append(AuditFinding(finding_id=f"F{len(self._findings)+1:03d}", category=action.action_type, severity="high", description=action.value, evidence=action.reasoning or ""))

    def _build_obs(self) -> Observation:
        doc_index = {k: ("[ACCESSED] " + v if k in self._docs_accessed else "[NOT ACCESSED]") for k, v in self._cfg["documents"].items()}
        return Observation(task_id=self.task_id, task_description=self._cfg["description"], ai_system=self._cfg["ai_system"], findings=self._findings, documents=doc_index, step=self._step_count, max_steps=self._cfg["max_steps"])
