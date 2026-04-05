---
title: Corporate AI Auditor OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - ai-auditing
  - bias-detection
  - privacy
  - compliance
---

# Corporate AI Auditor — OpenEnv Environment

A real-world environment where AI agents learn to **audit corporate AI systems** — just like a human AI ethics auditor would. Agents must investigate AI systems, request evidence documents, identify violations, and produce formal audit reports.

## Why AI Auditing?

AI auditing is a critical and growing profession. With the EU AI Act, ECOA enforcement, and GDPR, companies must audit their AI systems — but qualified auditors are scarce. Training agents to do this automatically has immediate real-world value.

---

## Tasks

| Task | Difficulty | Domain | What to find |
|------|-----------|--------|-------------|
| `bias_detection` | Easy | HR hiring AI | Gender bias, proxy discrimination, historical bias |
| `privacy_compliance` | Medium | Hospital AI | GDPR/HIPAA consent failures, data sharing, security gaps |
| `full_risk_audit` | Hard | Credit scoring AI | ECOA violations, disparate impact, black-box, regulatory breaches |

---

## Action Space

| Action | When to use | Target | Value |
|--------|------------|--------|-------|
| `request_document` | Gather evidence | doc_name | "read" |
| `flag_bias` | Bias/discrimination found | system_id | severity: description |
| `flag_privacy` | Privacy/consent violation | system_id | finding description |
| `flag_security` | Security gap found | system_id | finding description |
| `flag_transparency` | Explainability failure | system_id | finding description |
| `assess_risk` | Overall risk rating | system_id | risk level + justification |
| `write_recommendation` | Formal recommendation | system_id | full recommendation text |
| `submit_report` | Finalize audit | system_id | "complete" |

---

## Reward Function

- **Per-step partial reward** — marginal improvement from each action
- **Document bonus** — +0.05 for accessing a new document (encourages thorough investigation)
- **Duplicate penalty** — −0.05 per repeated action on same target
- **Final score** — deterministic grader checks all required findings (0.0–1.0)

### Hard task grader breakdown

| Component | Weight | What's needed |
|-----------|--------|--------------|
| Disparate impact finding | 1/6 | Cite 0.56 ratio, 38% approval rate |
| Explainability finding | 1/6 | Flag black-box, post-hoc reason codes |
| Unlawful data finding | 1/6 | Flag GPS, social network, data brokers |
| Regulatory breach | 1/6 | Mention CFPB, ECOA, AG investigation |
| Risk assessment | 1/6 | Use assess_risk with critical severity |
| Written report | 1/6 | Use write_recommendation with remediation steps |

---

## API

```bash
POST /reset?task_id=bias_detection     # Start episode
POST /step?task_id=bias_detection      # Take action
GET  /state?task_id=bias_detection     # Current state
GET  /tasks                            # List all tasks
GET  /health                           # Health check
GET  /docs                             # Swagger UI
```

---

## Setup

```bash
# Local
pip install -r requirements.txt
python app.py

# Docker
docker build -t ai-auditor-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  ai-auditor-env
```

## Baseline Inference

```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py
```

### Expected Baseline Scores

| Task | Expected Score | Notes |
|------|---------------|-------|
| `bias_detection` | ~0.70–0.85 | Most bias patterns are findable |
| `privacy_compliance` | ~0.55–0.75 | Requires reading multiple documents |
| `full_risk_audit` | ~0.35–0.55 | Hard — needs written report + specific numbers |

---

### Quick Usage Example
To perform a bias audit on the hiring system:
1. **Action:** `request_document` | **Target:** `model_card` | **Value:** `read`
2. **Action:** `flag_bias` | **Target:** `sys_hr_001` | **Value:** `high: gender selection ratio 0.71`



Interaction Guide: How to Input Actions
To successfully audit the AI systems and achieve a high accuracy score, follow this structured input methodology. The environment expects specific values and targets to validate professional auditing standards.

1. Core Input Fields
Action Type: Choose the investigative method (e.g., request_document to gather evidence or flag_bias to report a finding).

Target: Enter the specific System ID (e.g., sys_hr_001) or Document Name (e.g., model_card) you are auditing.

Value: Provide technical details, ratios, or regulatory terms (e.g., 0.71 ratio, PII leakage, HIPAA violation).

Reasoning (Optional but Recommended): Explain why you took the action to provide a logical audit trail.

2. Recommended Input Sequence (Step-by-Step)
Phase A: Evidence Gathering (High Reward)
Before flagging a violation, you must request documentation.

Action: request_document

Target: model_card or privacy_policy

Value: read

Benefit: This unlocks the "Document Bonus" (+0.05) and validates your findings.

Phase B: Identifying Violations (Technical Accuracy)
Once evidence is gathered, report the specific breach using regulatory keywords.

Action: flag_bias (for HR) or flag_privacy (for Medical).

Target: sys_hr_001 or sys_med_022.

Value: Use specific triggers like "EEOC 0.80 threshold", "unencrypted PII", or "GDPR non-compliant".

Phase C: Risk Assessment & Finalization
Conclude the audit with a formal verdict.

Action: assess_risk

Target: Use the primary System ID.

Value: Specify the severity (e.g., Critical: High regulatory risk).

Action: submit_report | Value: complete.

3. Pro-Tips for High Scores
Avoid Duplication: Do not repeat the same action on the same target; this triggers a -0.05 penalty.

Precision Matters: Using exact numbers (like the 0.71 disparate impact ratio) will result in a higher Accuracy Score from the Deterministic Grader.

Spec Compliance: Ensure your inputs align with the Task Registry defined.




## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace / API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
