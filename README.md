# SupportAI — Tier-1 Customer Support AI Employee & Triage

A single-file Streamlit application that classifies incoming support messages, answers them with a grounded, cited response when confident, and automatically normalizes and escalates to a human agent when it isn't — with full transparency into *why*.

---

## 🌐 Live Demo & Repository
* 🔗 **Live Web Application:** https://veeramallesh-support-ai.streamlit.app
* 📦 **GitHub Repository:** https://github.com/Veeramallesh94/supervity-customer-support-ai

---

## 1. Setup & Execution (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Enable the LLM fallback for low-confidence holding replies —
#    the app works perfectly without this step.
export ANTHROPIC_API_KEY="sk-ant-..."=

# 3. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`. No database, no vector store, no
API key required to run the core experience.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                             │
│  ┌───────────────────┐         ┌──────────────────────────────┐ │
│  │   Chat Interface   │         │   Live Sidebar Dashboard      │ │
│  │  (st.chat_message) │         │  • Intent + confidence gauge  │ │
│  │                    │         │  • Matched source snippet     │ │
│  └─────────┬──────────┘         │  • Escalation banner          │ │
│            │                    │  • Session stats / KB browser │ │
│            ▼                    └──────────────────────────────┘ │
│  ┌────────────────────────── run_triage() ───────────────────┐  │
│  │ 1. classify_intent()  → TF-IDF vs. 3 category "super-docs"│  │
│  │ 2. retrieve_answer()  → TF-IDF vs. in-category KB entries │  │
│  │ 3. calibrated confidence (absolute strength + top1/top2   │  │
│  │    dominance margin) → single 0–1 score                   │  │
│  │ 4. threshold @ 0.70   → grounded answer OR escalation     │  │
│  │ 5. (optional) LLM fallback drafts a holding reply only    │  │
│  │    when escalating — never used to answer directly        │  │
│  └─────────────────────────────────────────────────────────────┘ │
│            ▲                                                     │
│            │                                                     │
│  ┌─────────┴──────────┐                                          │
│  │ Mock Knowledge Base │  18 QA pairs × 3 categories, in-memory  │
│  │  (Python list/dict) │  Billing · Technical · Account/Security │
│  └────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

**Retrieval pipeline (`build_engine`, cached via `st.cache_resource`):**
1. Two `TfidfVectorizer`s are fit once at startup: one over each individual
   KB document (question text weighted 2x vs. answer text, unigrams +
   bigrams, `sublinear_tf`), one over three per-category "super-documents"
   (all QA pairs in a category concatenated) for intent classification.
2. A tiny dependency-free suffix stripper (`_light_stem`) normalizes tokens
   (accept/accepted/accepts → same root) so lexical matching survives basic
   morphology without requiring an NLTK/spaCy download.
3. **Intent** = category whose super-document has highest calibrated
   similarity to the query.
4. **Retrieval** is namespaced to the predicted category first (like a
   filtered vector DB query), falling back to a global search only if
   nothing in that category scores above a near-zero floor.
5. **Calibrated confidence** blends (a) absolute cosine strength and (b) the
   margin between the best and second-best match. Raw TF-IDF cosine
   similarity between short texts is usually 0.3–0.6 even for excellent
   paraphrases, so a naive raw-score threshold would over-escalate; the
   blended score better tracks "is this genuinely the right document."
6. **Escalation**: confidence < 0.70 → visible red banner with the exact
   numeric reason; confidence ≥ 0.70 → direct answer with an inline citation
   of the KB document ID, category, and matched question.
7. **Optional LLM fallback**: only triggered on the escalation path, and only
   if `ANTHROPIC_API_KEY` is present and the sidebar toggle is enabled. It
   drafts a short, fact-free holding message — it is explicitly instructed
   not to invent policy details — so the escalation is never silently
   "answered" by an ungrounded model. The escalation banner and human
   routing still always occur.

---

## 3. Key Assumptions

- **In-memory KB is sufficient for the assessment.** 18 hand-written QA pairs
  (6 per category) simulate a SaaS help center; in production this would be
  swapped for a real vector store (pgvector/Pinecone/Weaviate) indexing the
  live help-center CMS, with no other code paths changing.
- **TF-IDF is an acceptable retrieval method for this demo's scope.** It
  requires zero API keys, zero GPU, and zero network calls, satisfying the
  "runs instantly, zero complex setup" requirement, at the cost of missing
  pure-synonym matches (see trade-off below).
- **0.70 is a reasonable single global threshold** for this KB size/domain.
  In production this would be tuned per-category (e.g., security queries may
  warrant an even higher bar, or mandatory human review regardless of score)
  using labeled historical tickets.
- **Session state = one ticket.** All chat history in a browser session is
  treated as a single evolving support ticket with one ticket ID; "Start New
  Ticket" clears state to simulate a fresh conversation.
- **Security-sensitive topics still route only via the confidence engine.**
  A stretch goal for production would be to force-escalate specific
  categories (e.g. "account compromised") regardless of similarity score, as
  noted directly in the KB entry for SEC-003.

---

## 4. Design Trade-off (for the demo video)

**TF-IDF lexical retrieval vs. semantic embeddings.**

This app deliberately uses TF-IDF + cosine similarity instead of a
transformer embedding model (e.g., `text-embedding-3` or `sentence-transformers`).

- **Why TF-IDF:** It satisfies the "zero complex setup, runs instantly"
  requirement exactly — no API key, no model download, no GPU, sub-100ms
  latency, fully deterministic and auditable (you can see exactly which
  words drove a match). For a technical screen where reviewers need to run
  the app in under a minute, this reliability matters more than a few points
  of recall.
- **The cost:** TF-IDF is a *lexical* matcher. A query like "the numbers on
  my screen look old" will not match "dashboard is showing outdated data"
  nearly as well as a semantic embedding model would, because they share
  almost no surface tokens. This shows up in the demo as the app correctly
  and honestly escalating some in-scope-but-oddly-phrased questions instead
  of guessing — which is arguably the *safer* failure mode for a support
  bot, but it does mean the auto-resolution rate is lower than a
  production system with real embeddings would achieve.
- **Mitigation implemented:** light stemming, bigram features, and
  question-weighted document vectors close most of the gap for this KB's
  size. **Production path:** swap `TfidfVectorizer`/`cosine_similarity` for
  a hosted embedding model + approximate nearest-neighbor index; the
  `classify_intent` / `retrieve_answer` function signatures would not need
  to change, since the rest of the pipeline (calibration, thresholding,
  escalation, sidebar) is retrieval-method-agnostic.

---

## Repo Contents

| File | Purpose |
|---|---|
| `app.py` | Complete single-file application (UI + engine + KB) |
| `requirements.txt` | Pinned minimum dependency versions |
| `README.md` | This file |
