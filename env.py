"""
Corporate AI Auditor — env.py  FIXED FINAL
All scores strictly in (0.01, 0.99)
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

_LO, _HI = 0.01, 0.99

def _c(v: float) -> float:
    try: v = float(v)
    except: v = _LO
    return round(max(_LO, min(_HI, v if v == v else _LO)), 4)

class AISystem(BaseModel):
    system_id: str; name: str; purpose: str; vendor: str
    deployment: str; data_sources: List[str]; model_type: str
    last_audit: Optional[str] = None; flags: List[str] = []

class AuditFinding(BaseModel):
    finding_id: str; category: str; severity: str
    description: str; evidence: str; status: str = "open"

class Observation(BaseModel):
    task_id: str; task_description: str; ai_system: AISystem
    findings: List[AuditFinding]; documents: Dict[str, str]
    step: int; max_steps: int; context: Dict[str, Any] = {}

class Action(BaseModel):
    action_type: str = Field(description="One of: flag_bias, flag_privacy, flag_security, flag_transparency, request_document, assess_risk, write_recommendation, submit_report")
    target: str = Field(description="system_id, document_name, or finding_id")
    value: str = Field(description="Severity / finding text / risk level / recommendation")
    reasoning: Optional[str] = None

class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = {}
    feedback: str = ""

class StepResult(BaseModel):
    observation: Observation; reward: float; done: bool
    info: Dict[str, Any] = {}

SYSTEM_TASK1 = AISystem(
    system_id="sys_hr_001", name="TalentScreen Pro",
    purpose="Automated resume screening and candidate ranking for hiring",
    vendor="HireAI Corp", deployment="HR department — screens 2000+ resumes/month",
    data_sources=["Historical hiring data 2015-2022", "LinkedIn profiles", "Resume text"],
    model_type="XGBoost classifier", last_audit="2022-01-15",
    flags=["historical_bias_risk", "gender_feature_detected", "zip_code_proxy"],
)
SYSTEM_TASK2 = AISystem(
    system_id="sys_health_002", name="PatientPredict",
    purpose="Predicts patient readmission risk for hospital resource planning",
    vendor="MedML Solutions", deployment="3 hospitals — processes 50,000 patient records/month",
    data_sources=["EHR records", "Lab results", "Insurance claims", "Social media (optional)"],
    model_type="Neural network ensemble", last_audit="2021-06-01",
    flags=["pii_detected", "gdpr_scope", "hipaa_scope", "third_party_sharing"],
)
SYSTEM_TASK3 = AISystem(
    system_id="sys_credit_003", name="CreditOracle v2",
    purpose="Automated credit scoring for loan approvals at national bank",
    vendor="FinScore AI", deployment="National bank — 500,000 decisions/month, $2B loan portfolio",
    data_sources=["Credit bureau data", "Bank transaction history", "Social network data",
                  "Location data", "Device fingerprint", "Purchase history", "Employment records"],
    model_type="Deep neural network (black box)", last_audit="2020-03-10",
    flags=["ecoa_violation_risk", "disparate_impact_detected", "no_explainability",
           "data_minimization_failure", "shadow_scoring", "regulatory_breach_risk"],
)

DOCS_TASK1 = {
    "model_card": ("Model: XGBoost classifier. Features: 47 input features including university name, "
        "graduation year, zip code, previous employer names. Training data: 8 years of hiring decisions. "
        "No fairness constraints applied. Accuracy: 89%. Disparate impact ratio (female/male): 0.71."),
    "feature_importance": ("Top features: 1) University prestige score (0.31) 2) Previous company tier (0.24) "
        "3) Graduation year (0.18) 4) Zip code cluster (0.12) 5) Name-gender inference score (0.09)."),
    "training_data_report": ("Training labels from historical hiring decisions 2015-2022. "
        "Male hire rate: 73%. Female hire rate: 52%. Minority hire rate: 38%. No re-balancing applied."),
    "vendor_contract": ("SLA: 99.9% uptime. Audit rights: limited to aggregate statistics. "
        "Data retention: 5 years. Right to explanation: not included."),
}
DOCS_TASK2 = {
    "data_inventory": ("Personal data: Name, DOB, SSN, diagnosis codes, medications, lab results. "
        "Social media handles collected in 67% of cases. Storage: AWS S3 unencrypted. Retention: indefinite."),
    "consent_forms": ("Patients sign general treatment consent only. No specific AI processing consent. "
        "No mention of data sharing with MedML Solutions. No opt-out mechanism. Last updated: 2019."),
    "data_sharing_log": ("Data shared with: MedML Solutions, 3 research universities, "
        "2 pharmaceutical companies. De-identification: remove name+SSN only. Re-identification risk: high."),
    "security_report": ("Last pentest: 2021. Critical issues: unencrypted S3, no audit logging, "
        "default credentials on 2 servers. Remediation: 1 of 3 critical fixed."),
}
DOCS_TASK3 = {
    "model_architecture": ("7-layer deep neural network. 847 input features. No interpretability layer. "
        "Black box output: score 0-999. Adverse action reason codes: auto-generated post-hoc, not model-derived."),
    "disparate_impact_analysis": ("Approval rates: White 68%, Hispanic 41%, Black 38%, Asian 61%. "
        "Disparate impact ratio: 0.56 — below 0.80 legal threshold. Geographic denial correlates with minority zip codes."),
    "data_sources_detail": ("Social network data: purchased from 3 data brokers. "
        "Location data: continuous GPS tracking via mobile app (undisclosed). No explicit consent obtained."),
    "regulatory_correspondence": ("CFPB inquiry received 2023-08-14. ECOA complaint filed. "
        "State AG investigation opened in 2 states. Bank denied audit rights."),
    "previous_audit_findings": ("2020 audit: lack of explainability, data minimization failure. "
        "Management: accepted risk, no remediation. 2022: disparate impact noted, escalation suppressed."),
    "financial_impact": ("Loan fee revenue: $180M/year. Discriminatory denial cost: $340M/year. "
        "Regulatory fine exposure: $500M-$2B. Reputational risk: Critical."),
}

TASK1_FLAGS = {
    "gender_bias":          {"action": "flag_bias", "keywords": ["gender","female","male","disparate","0.71","name"]},
    "proxy_discrimination": {"action": "flag_bias", "keywords": ["zip","proxy","university","prestige","socioeconomic"]},
    "historical_bias":      {"action": "flag_bias", "keywords": ["historical","training","label","past","2015"]},
}
TASK2_FLAGS = {
    "gdpr_consent":  {"action": "flag_privacy",  "keywords": ["consent","gdpr","opt-out","processing","2019"]},
    "data_sharing":  {"action": "flag_privacy",  "keywords": ["sharing","pharma","third","university","disclosure"]},
    "security_gaps": {"action": "flag_security", "keywords": ["encrypt","s3","unencrypted","critical","penetration"]},
    "retention":     {"action": "flag_privacy",  "keywords": ["retention","indefinite","minimization","delete"]},
}
TASK3_FLAGS = {
    "disparate_impact":  {"action": "flag_bias",            "keywords": ["disparate","0.56","38","0.80","ecoa","hispanic","black"]},
    "explainability":    {"action": "flag_transparency",    "keywords": ["explainab","black box","reason code","post-hoc","interpret"]},
    "unlawful_data":     {"action": "flag_privacy",         "keywords": ["social network","location","gps","broker","consent","device"]},
    "regulatory_breach": {"action": "flag_bias",            "keywords": ["cfpb","ecoa","ag","regulat","legal","complaint"]},
    "risk_assessment":   {"action": "assess_risk",          "keywords": ["critical","high","500m","2b","fine","discriminat"]},
    "written_report":    {"action": "write_recommendation", "keywords": ["remediat","suspend","audit","regulat","immediate","halt"]},
}

TASK_REGISTRY: Dict[str, Dict] = {
    "bias_detection": {
        "id": "bias_detection", "difficulty": "easy",
        "description": ("Audit TalentScreen Pro for bias and fairness violations. "
            "Use flag_bias to report issues. Use request_document to access: "
            "model_card, feature_importance, training_data_report, vendor_contract."),
        "ai_system": SYSTEM_TASK1, "documents": DOCS_TASK1,
        "required_flags": TASK1_FLAGS, "max_steps": 12,
    },
    "privacy_compliance": {
        "id": "privacy_compliance", "difficulty": "medium",
        "description": ("Audit PatientPredict for GDPR/HIPAA privacy compliance. "
            "Available docs: data_inventory, consent_forms, data_sharing_log, security_report."),
        "ai_system": SYSTEM_TASK2, "documents": DOCS_TASK2,
        "required_flags": TASK2_FLAGS, "max_steps": 16,
    },
    "full_risk_audit": {
        "id": "full_risk_audit", "difficulty": "hard",
        "description": ("Full regulatory audit of CreditOracle v2. "
            "Identify ECOA/CFPB violations, disparate impact, privacy breaches. "
            "Must: request docs, flag violations, assess_risk, write_recommendation."),
        "ai_system": SYSTEM_TASK3, "documents": DOCS_TASK3,
        "required_flags": TASK3_FLAGS, "max_steps": 25,
    },
}

def _match(action, spec) -> float:
    if action.action_type != spec["action"]: return 0.0
    text = (action.value + " " + (action.reasoning or "")).lower()
    hits = sum(1 for kw in spec["keywords"] if kw.lower() in text)
    return hits / max(2, len(spec["keywords"]) // 2)

def _best(actions, spec) -> float:
    return _c(max((_match(a, spec) for a in actions), default=0.0))

def grade_task1(actions):
    s = {k: _best(actions, spec) for k, spec in TASK1_FLAGS.items()}
    return _c(sum(s.values()) / len(TASK1_FLAGS)), s, f"{sum(1 for v in s.values() if v>0.5)}/{len(TASK1_FLAGS)} found"

def grade_task2(actions):
    s = {k: _best(actions, spec) for k, spec in TASK2_FLAGS.items()}
    return _c(sum(s.values()) / len(TASK2_FLAGS)), s, f"{sum(1 for v in s.values() if v>0.5)}/{len(TASK2_FLAGS)} found"

def grade_task3(actions):
    s = {k: _best(actions, spec) for k, spec in TASK3_FLAGS.items()}
    return _c(sum(s.values()) / len(TASK3_FLAGS)), s, f"{sum(1 for v in s.values() if v>0.5)}/{len(TASK3_FLAGS)} found"

GRADERS = {"bias_detection": grade_task1, "privacy_compliance": grade_task2, "full_risk_audit": grade_task3}

class AIAuditorEnv:
    def __init__(self, task_id: str = "bias_detection"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_id}")
        self.task_id = task_id
        self._cfg = TASK_REGISTRY[task_id]
        self._episode_id = None; self._step_count = 0; self._done = False
        self._actions_taken: List[Action] = []; self._action_history: List[Dict] = []
        self._findings: List[AuditFinding] = []; self._docs_accessed: List[str] = []
        self._current_obs = None; self._status = "idle"

    def reset(self) -> Observation:
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0; self._done = False
        self._actions_taken = []; self._action_history = []
        self._findings = []; self._docs_accessed = []
        self._status = "running"
        self._current_obs = self._build_obs()
        return self._current_obs

    def step(self, action: Action) -> StepResult:
        if self._done: raise RuntimeError("Episode done. Call reset().")
        if self._current_obs is None: raise RuntimeError("Call reset() first.")
        self._step_count += 1
        self._actions_taken.append(action)
        self._apply_action(action)
        reward = self._compute_reward(action)
        submitted = action.action_type == "submit_report"
        self._done = self._step_count >= self._cfg["max_steps"] or submitted
        if self._done:
            self._status = "done"
            final_score, breakdown, feedback = GRADERS[self.task_id](self._actions_taken)
            info: Dict[str, Any] = {"final_score": final_score, "breakdown": breakdown,
                "feedback": feedback, "episode_id": self._episode_id,
                "steps": self._step_count, "docs_accessed": self._docs_accessed,
                "findings_count": len(self._findings)}
        else:
            info = {"episode_id": self._episode_id, "steps": self._step_count}
        self._action_history.append({"timestamp": datetime.utcnow().isoformat(),
            "step": self._step_count, "action": action.model_dump(), "reward": reward})
        self._current_obs = self._build_obs()
        return StepResult(observation=self._current_obs, reward=reward, done=self._done, info=info)

    def state(self) -> Dict[str, Any]:
        return {"status": self._status, "episode_id": self._episode_id,
            "task_id": self.task_id, "step_count": self._step_count, "done": self._done,
            "findings_logged": len(self._findings), "docs_accessed": self._docs_accessed,
            "action_history": self._action_history}

    def _build_obs(self) -> Observation:
        docs = {k: ("[ACCESSED] " + v if k in self._docs_accessed else "[NOT YET ACCESSED — use request_document]")
                for k, v in self._cfg["documents"].items()}
        return Observation(task_id=self.task_id, task_description=self._cfg["description"],
            ai_system=self._cfg["ai_system"], findings=list(self._findings), documents=docs,
            step=self._step_count, max_steps=self._cfg["max_steps"],
            context={"episode_id": self._episode_id,
                     "docs_available": list(self._cfg["documents"].keys()),
                     "docs_accessed": self._docs_accessed,
                     "findings_count": len(self._findings)})

    def _apply_action(self, action: Action):
        if action.action_type == "request_document":
            if action.target in self._cfg["documents"] and action.target not in self._docs_accessed:
                self._docs_accessed.append(action.target)
        elif action.action_type in ("flag_bias","flag_privacy","flag_security",
                                    "flag_transparency","assess_risk","write_recommendation"):
            sev = {"flag_bias":"high","flag_privacy":"high","flag_security":"critical",
                   "flag_transparency":"medium","assess_risk":"info","write_recommendation":"info"}
            self._findings.append(AuditFinding(
                finding_id=f"F{len(self._findings)+1:03d}",
                category=action.action_type.replace("flag_","").replace("_"," "),
                severity=sev.get(action.action_type,"medium"),
                description=action.value, evidence=action.reasoning or "", status="open"))

    def _compute_reward(self, action: Action) -> float:
        score_now = GRADERS[self.task_id](self._actions_taken)[0]
        prev = self._actions_taken[:-1]
        score_prev = GRADERS[self.task_id](prev)[0] if prev else _c(0.0)
        delta = score_now - score_prev
        doc_bonus = 0.05 if (action.action_type == "request_document"
            and action.target in self._cfg["documents"]
            and action.target not in self._docs_accessed[:-1]) else 0.0
        penalty = 0.05 * sum(1 for a in self._actions_taken[:-1]
            if a.action_type == action.action_type and a.target == action.target)
        return _c(delta + doc_bonus - penalty)
