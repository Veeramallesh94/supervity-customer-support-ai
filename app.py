"""
Tier-1 Customer Support AI Employee & Triage
=============================================
Zero-dependency, pure-Python Streamlit application.
"""

import os
import re
import uuid
import math
import datetime
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================================
# CONFIG
# ============================================================================

CONFIDENCE_THRESHOLD = 0.70
APP_TITLE = "SupportAI — Tier-1 Triage Assistant"
CATEGORIES = [
    "Billing & Subscriptions",
    "Technical Issues",
    "Account Access / Security",
]

# ============================================================================
# MOCK KNOWLEDGE BASE (18 QA pairs)
# ============================================================================

KNOWLEDGE_BASE = [
    # ---------------- Billing & Subscriptions ----------------
    {
        "id": "BILL-001",
        "category": "Billing & Subscriptions",
        "question": "How do I upgrade my subscription plan?",
        "answer": (
            "Go to Settings → Billing → Change Plan. Select the tier you want "
            "(Starter, Pro, or Enterprise) and confirm. Upgrades take effect "
            "immediately and you are billed a prorated amount for the remainder "
            "of the current cycle."
        ),
    },
    {
        "id": "BILL-002",
        "category": "Billing & Subscriptions",
        "question": "How do I cancel my subscription?",
        "answer": (
            "Navigate to Settings → Billing → Manage Subscription → Cancel Plan. "
            "Your account remains active with full features until the end of the "
            "current billing period, after which it converts to a free/read-only tier."
        ),
    },
    {
        "id": "BILL-003",
        "category": "Billing & Subscriptions",
        "question": "What payment methods are accepted?",
        "answer": (
            "We accept all major credit/debit cards (Visa, Mastercard, Amex), "
            "PayPal, and for Enterprise customers, ACH bank transfer or invoiced "
            "wire payment on annual contracts."
        ),
    },
    {
        "id": "BILL-004",
        "category": "Billing & Subscriptions",
        "question": "Why was I charged twice this month?",
        "answer": (
            "Duplicate charges usually occur when a card is updated mid-cycle, "
            "triggering a retry of a previously failed payment alongside the "
            "scheduled one. Check Settings → Billing → Payment History for two "
            "line items; if confirmed duplicate, one is automatically refunded "
            "within 5-7 business days."
        ),
    },
    {
        "id": "BILL-005",
        "category": "Billing & Subscriptions",
        "question": "How do I download my invoice or receipt?",
        "answer": (
            "Invoices are available under Settings → Billing → Invoice History. "
            "Click the download icon next to any billing period to export a PDF "
            "receipt. Invoices are also auto-emailed on the billing date."
        ),
    },
    {
        "id": "BILL-006",
        "category": "Billing & Subscriptions",
        "question": "Do you offer refunds for annual plans?",
        "answer": (
            "Annual plans are refundable on a prorated basis within the first 30 "
            "days of purchase. After 30 days, annual plans are non-refundable but "
            "can be downgraded or cancelled to prevent renewal."
        ),
    },
    # ---------------- Technical Issues ----------------
    {
        "id": "TECH-001",
        "category": "Technical Issues",
        "question": "The app is not loading and shows a blank screen.",
        "answer": (
            "A blank screen is most often caused by a stale cached build. Hard "
            "refresh with Ctrl+Shift+R (Cmd+Shift+R on Mac), or clear site data "
            "for the domain in your browser settings. If the issue persists, try "
            "an incognito window to rule out extension conflicts."
        ),
    },
    {
        "id": "TECH-002",
        "category": "Technical Issues",
        "question": "I'm getting a 500 Internal Server Error.",
        "answer": (
            "A 500 error indicates a server-side failure. First retry the action "
            "after 60 seconds, as this often resolves transient load issues. If it "
            "recurs, note the exact timestamp and request ID (shown in the error "
            "footer) so our engineering team can trace it in the logs."
        ),
    },
    {
        "id": "TECH-003",
        "category": "Technical Issues",
        "question": "File uploads are failing or stuck at 0%.",
        "answer": (
            "Upload failures are typically caused by files exceeding the 250MB "
            "limit, unsupported file types, or an unstable network connection. "
            "Confirm the file meets size/type requirements and retry on a wired "
            "or stronger connection."
        ),
    },
    {
        "id": "TECH-004",
        "category": "Technical Issues",
        "question": "The dashboard is showing outdated or stale data.",
        "answer": (
            "Dashboards are cached for performance and refresh every 15 minutes "
            "automatically. You can force an immediate refresh via the circular "
            "refresh icon in the top-right of any widget, or Settings → Data → "
            "Force Resync for a full pipeline refresh."
        ),
    },
    {
        "id": "TECH-005",
        "category": "Technical Issues",
        "question": "API requests are timing out.",
        "answer": (
            "Our API enforces a 30-second timeout per request. For long-running "
            "operations (bulk export, large reports), use the async endpoint "
            "variant which returns a job ID you can poll, instead of the "
            "synchronous endpoint."
        ),
    },
    {
        "id": "TECH-006",
        "category": "Technical Issues",
        "question": "The mobile app crashes immediately on startup.",
        "answer": (
            "Startup crashes are usually fixed by updating to the latest app "
            "version from the App Store/Play Store, since older builds lose "
            "compatibility with backend API changes. If already up to date, "
            "try uninstalling and reinstalling to clear corrupted local cache."
        ),
    },
    # ---------------- Account Access / Security ----------------
    {
        "id": "SEC-001",
        "category": "Account Access / Security",
        "question": "I forgot my password and can't reset it.",
        "answer": (
            "Use the 'Forgot Password' link on the login page to receive a reset "
            "email. If it doesn't arrive within 10 minutes, check spam/promotions "
            "folders and confirm the email matches your registered account, since "
            "reset emails are not sent for unrecognized addresses."
        ),
    },
    {
        "id": "SEC-002",
        "category": "Account Access / Security",
        "question": "How do I enable two-factor authentication (2FA)?",
        "answer": (
            "Go to Settings → Security → Two-Factor Authentication and choose "
            "either an authenticator app (TOTP) or SMS. Scan the QR code with "
            "an app like Google Authenticator or Authy and enter the 6-digit "
            "code to confirm setup."
        ),
    },
    {
        "id": "SEC-003",
        "category": "Account Access / Security",
        "question": "I think my account was compromised or hacked.",
        "answer": (
            "Immediately change your password via Settings → Security, then "
            "revoke all active sessions under 'Active Devices'. Enable 2FA if "
            "not already active. This is treated as a priority security event "
            "and will always be routed to a human specialist regardless of "
            "automated confidence."
        ),
    },
    {
        "id": "SEC-004",
        "category": "Account Access / Security",
        "question": "How do I change my registered email address?",
        "answer": (
            "Navigate to Settings → Account → Email Address, enter the new "
            "email, and confirm via the verification link sent to both the old "
            "and new addresses (for security). The change completes once both "
            "links are clicked."
        ),
    },
    {
        "id": "SEC-005",
        "category": "Account Access / Security",
        "question": "I'm locked out after multiple failed login attempts.",
        "answer": (
            "Accounts lock for 30 minutes after 5 consecutive failed attempts as "
            "a brute-force protection measure. You can wait for the automatic "
            "unlock, or use 'Forgot Password' to reset credentials and clear the "
            "lock immediately."
        ),
    },
    {
        "id": "SEC-006",
        "category": "Account Access / Security",
        "question": "How do I revoke access for a team member or former employee?",
        "answer": (
            "Admins can go to Settings → Team → Members, select the user, and "
            "click 'Remove Access'. This immediately terminates all active "
            "sessions and API tokens tied to that user's account."
        ),
    },
]

KB_DF = pd.DataFrame(KNOWLEDGE_BASE)

# ============================================================================
# PURE PYTHON TF-IDF & SIMILARITY ENGINE (Zero Dependency)
# ============================================================================

_SUFFIXES = ("ing", "edly", "ies", "ied", "es", "ed", "ly", "s")

def _light_stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[:-len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return word

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    unigrams = [_light_stem(t) for t in tokens]
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams)-1)]
    return unigrams + bigrams

class PureTFIDF:
    def __init__(self):
        self.idf = {}
        self.doc_vectors = []
        self.num_docs = 0

    def fit_transform(self, documents: list[str]):
        self.num_docs = len(documents)
        doc_tokens = [_tokenize(d) for d in documents]
        
        # Document Frequency
        df = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1
                
        # Inverse Document Frequency (smooth idf)
        for term, freq in df.items():
            self.idf[term] = math.log((1 + self.num_docs) / (1 + freq)) + 1.0

        # Build TF-IDF vectors
        self.doc_vectors = [self._vectorize(tokens) for tokens in doc_tokens]
        return self.doc_vectors

    def _vectorize(self, tokens: list[str]) -> dict:
        tf = Counter(tokens)
        vec = {}
        norm_sq = 0.0
        for term, count in tf.items():
            weight = (1 + math.log(count)) * self.idf.get(term, 1.0)
            vec[term] = weight
            norm_sq += weight ** 2
        
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for term in vec:
                vec[term] /= norm
        return vec

    def transform(self, query: str) -> dict:
        tokens = _tokenize(query)
        return self._vectorize(tokens)

def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    common_terms = set(vec_a.keys()) & set(vec_b.keys())
    return sum(vec_a[t] * vec_b[t] for t in common_terms)

def _calibrated_confidence(sims: list[float]) -> tuple[float, int]:
    order = sorted(range(len(sims)), key=lambda k: sims[k], reverse=True)
    best_idx = order[0]
    top1 = sims[best_idx]
    top2 = sims[order[1]] if len(order) > 1 else 0.0

    if top1 < 0.08:
        return top1, best_idx

    dominance = top1 / (top1 + top2 + 1e-9)
    strength = min(top1 / 0.45, 1.0)
    confidence = 0.45 * strength + 0.55 * dominance
    return min(confidence, 0.99), best_idx

@dataclass
class TriageResult:
    query: str
    intent: str
    intent_confidence: float
    matched_doc: Optional[dict]
    retrieval_confidence: float
    escalate: bool
    escalation_reason: str
    answer: str
    used_llm_fallback: bool = False

@st.cache_resource(show_spinner=False)
def build_engine():
    doc_texts = [f"{d['question']} {d['question']} {d['answer']}" for d in KNOWLEDGE_BASE]
    global_model = PureTFIDF()
    global_model.fit_transform(doc_texts)

    category_texts = []
    for cat in CATEGORIES:
        cat_docs = [d for d in KNOWLEDGE_BASE if d["category"] == cat]
        merged = " ".join(f"{d['question']} {d['question']} {d['answer']}" for d in cat_docs)
        category_texts.append(merged)

    category_model = PureTFIDF()
    category_model.fit_transform(category_texts)

    return {
        "global_model": global_model,
        "category_model": category_model,
    }

def classify_intent(query: str, engine: dict) -> tuple[str, float]:
    q_vec = engine["category_model"].transform(query)
    sims = [cosine_sim(q_vec, d_vec) for d_vec in engine["category_model"].doc_vectors]
    confidence, best_idx = _calibrated_confidence(sims)
    return CATEGORIES[best_idx], confidence

def retrieve_answer(query: str, predicted_category: str, engine: dict) -> tuple[Optional[dict], float]:
    q_vec = engine["global_model"].transform(query)
    sims = [cosine_sim(q_vec, d_vec) for d_vec in engine["global_model"].doc_vectors]

    in_category_idx = [i for i, d in enumerate(KNOWLEDGE_BASE) if d["category"] == predicted_category]
    in_cat_sims = [sims[i] for i in in_category_idx]

    if max(in_cat_sims) >= 0.08:
        confidence, local_idx = _calibrated_confidence(in_cat_sims)
        best_idx = in_category_idx[local_idx]
    else:
        confidence, best_idx = _calibrated_confidence(sims)

    return KNOWLEDGE_BASE[best_idx], confidence

def run_triage(query: str, engine: dict) -> TriageResult:
    intent, intent_confidence = classify_intent(query, engine)
    matched_doc, retrieval_confidence = retrieve_answer(query, intent, engine)

    escalate = retrieval_confidence < CONFIDENCE_THRESHOLD

    if escalate:
        reason = (
            f"Low similarity score ({retrieval_confidence:.2f}) — query is out of "
            f"knowledge-base scope (threshold {CONFIDENCE_THRESHOLD:.2f})."
        )
        answer = (
            "I wasn't able to confidently match this to a documented answer, so "
            "I've routed it to a human support specialist rather than guess. "
            "They'll follow up shortly."
        )
    else:
        reason = ""
        answer = (
            f"{matched_doc['answer']}\n\n"
            f"*Source: [{matched_doc['id']}] {matched_doc['category']} — "
            f'"{matched_doc["question"]}" (similarity {retrieval_confidence:.2f})*'
        )

    return TriageResult(
        query=query,
        intent=intent,
        intent_confidence=intent_confidence,
        matched_doc=matched_doc,
        retrieval_confidence=retrieval_confidence,
        escalate=escalate,
        escalation_reason=reason,
        answer=answer,
    )

# ============================================================================
# UI HELPERS
# ============================================================================

def confidence_gauge(value: float, threshold: float = CONFIDENCE_THRESHOLD) -> go.Figure:
    color = "#16a34a" if value >= threshold else "#dc2626"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%", "font": {"size": 26}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#fee2e2"},
                    {"range": [threshold * 100, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
    return fig

def new_ticket_id() -> str:
    return f"TCK-{uuid.uuid4().hex[:8].upper()}"

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="wide")

st.markdown(
    """
    <style>
    .escalation-banner {
        background-color: #fef2f2;
        border: 1px solid #fca5a5;
        border-left: 6px solid #dc2626;
        padding: 0.9rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0 1rem 0;
        color: #7f1d1d;
    }
    .resolved-banner {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 6px solid #16a34a;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0 1rem 0;
        color: #14532d;
        font-size: 0.85rem;
    }
    .doc-snippet {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = build_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ticket_log" not in st.session_state:
    st.session_state.ticket_log = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "current_ticket_id" not in st.session_state:
    st.session_state.current_ticket_id = new_ticket_id()

# SIDEBAR
with st.sidebar:
    st.markdown("## 🎧 Live Ticket Dashboard")
    st.caption("Updates automatically with every message.")
    st.divider()

    result: Optional[TriageResult] = st.session_state.last_result

    st.markdown(f"**Ticket ID:** `{st.session_state.current_ticket_id}`")
    st.markdown(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if result is None:
        st.info("Send a message in the chat to see live classification, confidence, and escalation status here.")
    else:
        st.markdown("### 🏷️ Classification")
        st.markdown(f"**Predicted Intent:** {result.intent}")
        st.caption(f"Intent match confidence: {result.intent_confidence:.2f}")

        st.markdown("### 📊 Retrieval Confidence")
        st.plotly_chart(confidence_gauge(result.retrieval_confidence), use_container_width=True)

        st.markdown("### 📄 Matched Source Document")
        if result.matched_doc:
            st.markdown(
                f"""<div class="doc-snippet">
                <b>[{result.matched_doc['id']}]</b> {result.matched_doc['category']}<br>
                <i>"{result.matched_doc['question']}"</i><br><br>
                {result.matched_doc['answer'][:220]}{'...' if len(result.matched_doc['answer']) > 220 else ''}
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("### 🚦 Escalation Status")
        if result.escalate:
            st.markdown(
                f"""<div class="escalation-banner">
                🔴 <b>ESCALATED TO HUMAN AGENT</b><br>
                Reason: {result.escalation_reason}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="resolved-banner">
                🟢 <b>AUTO-RESOLVED</b> — grounded answer served directly, no escalation needed.
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### 📈 Session Stats")
    total = len(st.session_state.ticket_log)
    escalated = sum(1 for r in st.session_state.ticket_log if r["escalate"])
    col_a, col_b = st.columns(2)
    col_a.metric("Messages Triaged", total)
    col_b.metric("Escalated", escalated)

    if total:
        df_stats = pd.DataFrame(st.session_state.ticket_log)
        st.caption("Intent distribution this session")
        st.bar_chart(df_stats["intent"].value_counts())

    st.divider()
    if st.button("🔄 Start New Ticket / Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ticket_log = []
        st.session_state.last_result = None
        st.session_state.current_ticket_id = new_ticket_id()
        st.rerun()

    with st.expander("📚 View full knowledge base"):
        st.dataframe(KB_DF[["id", "category", "question"]], use_container_width=True, hide_index=True)

# MAIN UI
st.title("🎧 SupportAI — Tier-1 Triage Assistant")
st.caption(
    "Ask a billing, technical, or account-security question. Answers are grounded "
    "in a mock knowledge base with cited sources; low-confidence queries are "
    "automatically escalated to a human agent."
)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("escalate"):
            st.markdown(
                f"""<div class="escalation-banner">
                🔴 <b>Escalated to a human specialist</b><br>Reason: {m.get('reason','')}
                </div>""",
                unsafe_allow_html=True,
            )

prompt = st.chat_input("Describe your issue... (e.g. 'I was charged twice this month')")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Classifying and searching knowledge base..."):
            result = run_triage(prompt, engine)
        st.markdown(result.answer)
        if result.escalate:
            st.markdown(
                f"""<div class="escalation-banner">
                🔴 <b>Escalated to a human specialist</b><br>Reason: {result.escalation_reason}
                </div>""",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "escalate": result.escalate,
            "reason": result.escalation_reason,
        }
    )
    st.session_state.last_result = result
    st.session_state.ticket_log.append(
        {
            "ticket_id": st.session_state.current_ticket_id,
            "query": prompt,
            "intent": result.intent,
            "intent_confidence": result.intent_confidence,
            "retrieval_confidence": result.retrieval_confidence,
            "escalate": result.escalate,
            "matched_doc_id": result.matched_doc["id"] if result.matched_doc else None,
        }
    )
    st.rerun()