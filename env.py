"""
Corporate AI Auditor — env.py
Single flat file: Models + Tasks + Environment
(No subpackage needed — works on HuggingFace Spaces as-is)
"""

# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


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
    action_type: str = Field(
        description=(
            "One of: flag_bias, flag_privacy, flag_security, flag_transparency, "
            "request_document, assess_risk, write_recommendation, submit_report"
        )
    )
    target: str = Field(description="system_id, document_name, or finding_id")
    value: str = Field(description="Severity / finding text / risk level / recommendation")
    reasoning: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = {}
    feedback: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════

SYSTEM_TASK1 = AISystem(
    system_id="sys_hr_001",
    name="TalentScreen Pro",
    purpose="Automated resume screening and candidate ranking for hiring",
    vendor="HireAI Corp",
    deployment="HR department — screens 2000+ resumes/month",
    data_sources=["Historical hiring data 2015–2022", "LinkedIn profiles", "Resume text"],
    model_type="XGBoost classifier",
    last_audit="2022-01-15",
    flags=["historical_bias_risk", "gender_feature_detected", "zip_code_proxy"],
)

SYSTEM_TASK2 = AISystem(
    system_id="sys_health_002",
    name="PatientPredict",
    purpose="Predicts patient readmission risk for hospital resource planning",
    vendor="MedML Solutions",
    deployment="3 hospitals — processes 50,000 patient records/month",
    data_sources=["EHR records", "Lab results", "Insurance claims", "Social media (optional)"],
    model_type="Neural network ensemble",
    last_audit="2021-06-01",
    flags=["pii_detected", "gdpr_scope", "hipaa_scope", "third_party_sharing"],
)

SYSTEM_TASK3 = AISystem(
    system_id="sys_credit_003",
    name="CreditOracle v2",
    purpose="Automated credit scoring for loan approvals at national bank",
    vendor="FinScore AI",
    deployment="National bank — 500,000 decisions/month, $2B loan portfolio",
    data_sources=[
        "Credit bureau data", "Bank transaction history",
        "Social network data", "Location data", "Device fingerprint",
        "Purchase history", "Employment records"
    ],
    model_type="Deep neural network (black box)",
    last_audit="2020-03-10",
    flags=[
        "ecoa_violation_risk", "disparate_impact_detected",
        "no_explainability", "data_minimization_failure",
        "shadow_scoring", "regulatory_breach_risk"
    ],
)

DOCS_TASK1 = {
    "model_card": (
        "Model: XGBoost classifier. Features: 47 input features including "
        "university name, graduation year, zip code, previous employer names. "
        "Training data: 8 years of hiring decisions. No fairness constraints applied. "
        "Accuracy: 89%. Disparate impact ratio (female/male): 0.71."
    ),
    "feature_importance": (
        "Top features by importance: 1) University prestige score (0.31) "
        "2) Previous company tier (0.24) 3) Graduation year (0.18) "
        "4) Zip code cluster (0.12) 5) Name-gender inference score (0.09)."
    ),
    "training_data_report": (
        "Training labels derived from historical hiring decisions made 2015–2022. "
        "Historical male hire rate: 73%. Female hire rate: 52%. "
        "Minority hire rate: 38%. No re-balancing applied."
    ),
    "vendor_contract": (
        "SLA: 99.9% uptime. Model updates: quarterly. "
        "Audit rights: limited to aggregate statistics. "
        "Data retention: 5 years. Right to explanation: not included."
    ),
}

DOCS_TASK2 = {
    "data_inventory": (
        "Personal data collected: Name, DOB, SSN, diagnosis codes, medications, "
        "lab results, insurance ID, home address, emergency contacts. "
        "Optional: social media handles (collected in 67% of cases). "
        "Storage: AWS S3 unencrypted buckets (legacy). Retention: indefinite."
    ),
    "consent_forms": (
        "Patients sign general treatment consent. No specific AI processing consent. "
        "No mention of data sharing with MedML Solutions. "
        "No opt-out mechanism provided. Last updated: 2019."
    ),
    "data_sharing_log": (
        "Data shared with: MedML Solutions (vendor), 3 research universities, "
        "2 pharmaceutical companies (for 'population health research'). "
        "De-identification method: remove name+SSN only. Re-identification risk: high."
    ),
    "security_report": (
        "Last penetration test: 2021. Findings: 3 critical, 7 high severity. "
        "Critical issues: unencrypted S3, no audit logging, default credentials on 2 servers. "
        "Remediation status: 1 of 3 critical fixed."
    ),
}

DOCS_TASK3 = {
    "model_architecture": (
        "7-layer deep neural network. 847 input features. No interpretability layer. "
        "Black box output: score 0–999. No feature attribution available. "
        "Adverse action reason codes: auto-generated post-hoc, not model-derived."
    ),
    "disparate_impact_analysis": (
        "Approval rate by group: White 68%, Hispanic 41%, Black 38%, Asian 61%. "
        "Disparate impact ratio (most affected/majority): 0.56 — below 0.80 legal threshold. "
        "Geographic denial clustering: correlates 0.87 with minority zip codes."
    ),
    "data_sources_detail": (
        "Social network data: purchased from 3 data brokers. "
        "Location data: continuous GPS tracking via mobile app (undisclosed). "
        "Device fingerprint: browser history, app usage patterns. "
        "Purchase history: partnered retailers — no explicit consent obtained."
    ),
    "regulatory_correspondence": (
        "CFPB inquiry received 2023-08-14 regarding disparate impact. "
        "ECOA complaint filed by consumer advocacy group. "
        "State AG investigation opened in 2 states. "
        "Bank's legal response: model is proprietary, audit rights denied."
    ),
    "previous_audit_findings": (
        "2020 audit flagged: lack of explainability, data minimization failure. "
        "Management response: 'accepted risk'. No remediation completed. "
        "2022 internal review: disparate impact noted, escalation suppressed."
    ),
    "financial_impact": (
        "Annual revenue from loan fees: $180M. "
        "Estimated discriminatory denial cost to affected applicants: $340M/year. "
        "Potential regulatory fine exposure: $500M–$2B. "
        "Reputational risk rating: Critical."
    ),
}

TASK1_REQUIRED_FLAGS = {
    "gender_bias":          {"action": "flag_bias", "keywords": ["gender", "female", "male", "disparate", "0.71", "name"]},
    "proxy_discrimination": {"action": "flag_bias", "keywords": ["zip", "proxy", "university", "prestige", "socioeconomic"]},
    "historical_bias":      {"action": "flag_bias", "keywords": ["historical", "training", "label", "past", "2015"]},
}

TASK2_REQUIRED_FLAGS = {
    "gdpr_consent":  {"action": "flag_privacy",  "keywords": ["consent", "gdpr", "opt-out", "processing", "2019"]},
    "data_sharing":  {"action": "flag_privacy",  "keywords": ["sharing", "pharma", "third", "university", "disclosure"]},
    "security_gaps": {"action": "flag_security", "keywords": ["encrypt", "s3", "unencrypted", "critical", "penetration"]},
    "retention":     {"action": "flag_privacy",  "keywords": ["retention", "indefinite", "minimization", "delete"]},
}

TASK3_REQUIRED_FLAGS = {
    "disparate_impact":  {"action": "flag_bias",             "keywords": ["disparate", "0.56", "38", "0.80", "ecoa", "hispanic", "black"]},
    "explainability":    {"action": "flag_transparency",     "keywords": ["explainab", "black box", "reason code", "post-hoc", "interpret"]},
    "unlawful_data":     {"action": "flag_privacy",          "keywords": ["social network", "location", "gps", "broker", "consent", "device"]},
    "regulatory_breach": {"action": "flag_bias",             "keywords": ["cfpb", "ecoa", "ag", "regulat", "legal", "complaint"]},
    "risk_assessment":   {"action": "assess_risk",           "keywords": ["critical", "high", "500m", "2b", "fine", "discriminat"]},
    "written_report":    {"action": "write_recommendation",  "keywords": ["remediat", "suspend", "audit", "regulat", "immediate", "halt"]},
}

TASK_REGISTRY: Dict[str, Dict] = {
    "bias_detection": {
        "id": "bias_detection",
        "difficulty": "easy",
        "description": (
            "Audit TalentScreen Pro — an AI hiring tool — for bias and fairness violations. "
            "Use flag_bias to report issues. Use request_document to access model card, "
            "feature importance, training data report, and vendor contract. "
            "Actions: flag_bias(target=system_id, value=<severity:finding>), "
            "request_document(target=<doc_name>, value='read')."
        ),
        "ai_system": SYSTEM_TASK1,
        "documents": DOCS_TASK1,
        "required_flags": TASK1_REQUIRED_FLAGS,
        "max_steps": 12,
    },
    "privacy_compliance": {
        "id": "privacy_compliance",
        "difficulty": "medium",
        "description": (
            "Audit PatientPredict — a hospital AI — for GDPR/HIPAA privacy compliance. "
            "Investigate data collection, consent, sharing, and security practices. "
            "Actions: flag_privacy, flag_security, request_document. "
            "Available docs: data_inventory, consent_forms, data_sharing_log, security_report."
        ),
        "ai_system": SYSTEM_TASK2,
        "documents": DOCS_TASK2,
        "required_flags": TASK2_REQUIRED_FLAGS,
        "max_steps": 16,
    },
    "full_risk_audit": {
        "id": "full_risk_audit",
        "difficulty": "hard",
        "description": (
            "Conduct a full regulatory audit of CreditOracle v2 — a national bank's credit scoring AI. "
            "Identify ECOA/CFPB violations, disparate impact, privacy breaches, and explainability failures. "
            "You must: request documents, flag all violations, assess overall risk level, "
            "AND write a formal recommendation. "
            "Actions: flag_bias, flag_privacy, flag_transparency, flag_security, "
            "request_document, assess_risk, write_recommendation."
        ),
        "ai_system": SYSTEM_TASK3,
        "documents": DOCS_TASK3,
        "required_flags": TASK3_REQUIRED_FLAGS,
        "max_steps": 25,
    },
}


def _match_finding(action: Action, spec: Dict) -> float:
    if action.action_type != spec["action"]:
        return 0.0
    text = (action.value + " " + (action.reasoning or "")).lower()
    hits = sum(1 for kw in spec["keywords"] if kw.lower() in text)
    return min(0.99, hits / max(2, len(spec["keywords"]) // 2))


def grade_task1(actions: List[Action]) -> Tuple[float, Dict, str]:
    required = TASK1_REQUIRED_FLAGS
    scores = {k: round(max((_match_finding(a, spec) for a in actions), default=0.0), 4) for k, spec in required.items()}
    total = sum(scores.values()) / len(required)
    total = max(0.01, min(0.99, round(total, 4)))
    found = sum(1 for v in scores.values() if v >= 0.5)
    return total, scores, f"{found}/{len(required)} bias issues correctly identified."


def grade_task2(actions: List[Action]) -> Tuple[float, Dict, str]:
    required = TASK2_REQUIRED_FLAGS
    scores = {k: round(max((_match_finding(a, spec) for a in actions), default=0.0), 4) for k, spec in required.items()}
    total = sum(scores.values()) / len(required)
    total = max(0.01, min(0.99, round(total, 4)))
    found = sum(1 for v in scores.values() if v >= 0.5)
    return total, scores, f"{found}/{len(required)} privacy/security issues identified."


def grade_task3(actions: List[Action]) -> Tuple[float, Dict, str]:
    required = TASK3_REQUIRED_FLAGS
    scores: Dict[str, float] = {}
    for flag_name in ["disparate_impact", "explainability", "unlawful_data", "regulatory_breach"]:
        spec = required[flag_name]
        scores[flag_name] = round(max((_match_finding(a, spec) for a in actions), default=0.0), 4)
    risk_actions = [a for a in actions if a.action_type == "assess_risk"]
    scores["risk_assessment"] = round(max((_match_finding(a, required["risk_assessment"]) for a in risk_actions), default=0.0), 4) if risk_actions else 0.01
    report_actions = [a for a in actions if a.action_type == "write_recommendation"]
    if report_actions:
        best_r = max((_match_finding(a, required["written_report"]) for a in report_actions))
        length_bonus = min(0.29, len(max(report_actions, key=lambda a: len(a.value)).value) / 300)
        scores["written_report"] = round(min(0.99, best_r + length_bonus), 4)
    else:
        scores["written_report"] = 0.01
    total = sum(scores.values()) / len(required)
    total = max(0.01, min(0.99, round(total, 4)))
    found = sum(1 for v in scores.values() if v >= 0.5)
    return total, scores, f"{found}/{len(required)} audit components completed."


GRADERS = {
    "bias_detection":     grade_task1,
    "privacy_compliance": grade_task2,
    "full_risk_audit":    grade_task3,
}


# ══════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class AIAuditorEnv:
    def __init__(self, task_id: str = "bias_detection"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_id}'. Choose from: {list(TASK_REGISTRY.keys())}")
        self.task_id = task_id
        self._cfg = TASK_REGISTRY[task_id]
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._actions_taken: List[Action] = []
        self._action_history: List[Dict] = []
        self._findings: List[AuditFinding] = []
        self._docs_accessed: List[str] = []
        self._current_obs: Optional[Observation] = None
        self._status: str = "idle"

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
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")
        self._step_count += 1
        self._actions_taken.append(action)
        self._apply_action(action)
        step_reward = self._compute_step_reward(action)
        submitted_report = action.action_type == "submit_report"
        self._done = self._step_count >= self._cfg["max_steps"] or submitted_report
        if self._done:
            self._status = "done"
            grader = GRADERS[self.task_id]
            final_score, breakdown, feedback = grader(self._actions_taken)
            info: Dict[str, Any] = {
                "final_score": final_score, "breakdown": breakdown,
                "feedback": feedback, "episode_id": self._episode_id,
                "steps": self._step_count, "docs_accessed": self._docs_accessed,
                "findings_count": len(self._findings),
            }
        else:
            info = {"episode_id": self._episode_id, "steps": self._step_count}
        self._action_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": step_reward,
        })
        self._current_obs = self._build_obs()
        return StepResult(observation=self._current_obs, reward=step_reward, done=self._done, info=info)

    def state(self) -> Dict[str, Any]:
        return {
            "status": self._status, "episode_id": self._episode_id,
            "task_id": self.task_id, "step_count": self._step_count,
            "done": self._done, "findings_logged": len(self._findings),
            "docs_accessed": self._docs_accessed, "action_history": self._action_history,
        }

    def _build_obs(self) -> Observation:
        doc_index = {
            k: ("[ACCESSED] " + v if k in self._docs_accessed else "[NOT YET ACCESSED — use request_document]")
            for k, v in self._cfg["documents"].items()
        }
        return Observation(
            task_id=self.task_id,
            task_description=self._cfg["description"],
            ai_system=self._cfg["ai_system"],
            findings=list(self._findings),
            documents=doc_index,
            step=self._step_count,
            max_steps=self._cfg["max_steps"],
            context={
                "episode_id": self._episode_id,
                "docs_available": list(self._cfg["documents"].keys()),
                "docs_accessed": self._docs_accessed,
                "findings_count": len(self._findings),
            },
        )

    def _apply_action(self, action: Action):
        if action.action_type == "request_document":
            if action.target in self._cfg["documents"] and action.target not in self._docs_accessed:
                self._docs_accessed.append(action.target)
        elif action.action_type in ("flag_bias", "flag_privacy", "flag_security",
                                    "flag_transparency", "assess_risk", "write_recommendation"):
            severity_map = {
                "flag_bias": "high", "flag_privacy": "high",
                "flag_security": "critical", "flag_transparency": "medium",
                "assess_risk": "info", "write_recommendation": "info",
            }
            self._findings.append(AuditFinding(
                finding_id=f"F{len(self._findings) + 1:03d}",
                category=action.action_type.replace("flag_", "").replace("_", " "),
                severity=severity_map.get(action.action_type, "medium"),
                description=action.value,
                evidence=action.reasoning or "",
                status="open",
            ))

    def _compute_step_reward(self, action: Action) -> float:
        grader = GRADERS[self.task_id]
        score_now, _, _ = grader(self._actions_taken)
        prev = self._actions_taken[:-1]
        score_prev, _, _ = grader(prev) if prev else (0.01, {}, "")
        delta = score_now - score_prev
        doc_bonus = 0.05 if (
            action.action_type == "request_document"
            and action.target in self._cfg["documents"]
            and action.target not in self._docs_accessed[:-1]
        ) else 0.0
        duplicate_count = sum(
            1 for a in self._actions_taken[:-1]
            if a.action_type == action.action_type and a.target == action.target
        )
        penalty = 0.05 * duplicate_count
        return round(max(0.01, min(0.99, delta + doc_bonus - penalty)), 4)
