"""
Unit Tests for Customer Support AI Triage Engine
================================================
Verifies classification accuracy, grounded retrieval, and escalation logic.
"""

import pytest
from app import classify_intent, retrieve_answer, build_engine, CONFIDENCE_THRESHOLD

@pytest.fixture(scope="module")
def engine():
    return build_engine()

def test_intent_classification_billing(engine):
    intent, conf = classify_intent("How do I upgrade my billing tier to pro?", engine)
    assert intent == "Billing & Subscriptions"
    assert conf > 0.40

def test_intent_classification_technical(engine):
    intent, conf = classify_intent("My mobile application is crashing on launch", engine)
    assert intent == "Technical Issues"
    assert conf > 0.40

def test_grounded_retrieval_success(engine):
    doc, conf = retrieve_answer("download my invoice pdf receipt", "Billing & Subscriptions", engine)
    assert doc is not None
    assert doc["id"] == "BILL-005"
    assert conf >= CONFIDENCE_THRESHOLD

def test_out_of_scope_escalation(engine):
    doc, conf = retrieve_answer("How do I bake a chocolate cake?", "Technical Issues", engine)
    assert conf < CONFIDENCE_THRESHOLD