import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re


class SupportAgent:
    def __init__(self, historical_data_path):
        """
        historical_data_path: CSV containing 'user_text', 'historical_brand_response', 'label_intent'
        """
        self.df = pd.read_csv(historical_data_path)

        # Intent labels
        self.intents = self.df['label_intent'].unique().tolist()

        # ── 1. Train Intent Classifier ────────────────────────────────────────
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True
        )
        self.X_train = self.vectorizer.fit_transform(self.df['user_text'])
        self.y_train = self.df['label_intent']

        self.intent_classifier = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=5.0,
            solver='lbfgs'
        )
        self.intent_classifier.fit(self.X_train, self.y_train)

    # ─────────────────────────────────────────────────────────────────────────
    def classify_intent(self, text):
        """Classify incoming customer message into an intent with confidence."""
        X_test = self.vectorizer.transform([text])
        intent = self.intent_classifier.predict(X_test)[0]
        proba = self.intent_classifier.predict_proba(X_test)[0]
        confidence = float(np.max(proba))

        # Build top-3 intent breakdown for transparency
        classes = self.intent_classifier.classes_
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [(classes[i], round(float(proba[i]), 3)) for i in top3_idx]

        return intent, confidence, top3

    # ─────────────────────────────────────────────────────────────────────────
    def route_decision(self, text, intent, confidence):
        """
        Decide Auto-handle vs Escalate with a human-readable reason.
        Rules (in priority order):
          1. Very low confidence → escalate
          2. Urgent language detected → escalate regardless of intent
          3. Sensitive intents → escalate
          4. Everything else → auto-handle
        """
        text_lower = text.lower()

        # Rule 1: low confidence
        if confidence < 0.40:
            return (
                "Escalate",
                "Confidence below threshold — unclear intent. Routing to a human agent to ensure accuracy."
            )

        # Rule 2: detect urgency / distress signals
        urgent_keywords = [
            "urgent", "asap", "immediately", "emergency", "critical",
            "stolen", "lost all", "data loss", "unauthorized", "hacked",
            "scam", "fraud", "cannot work", "costing me"
        ]
        if any(kw in text_lower for kw in urgent_keywords):
            return (
                "Escalate",
                "Urgent or high-stakes language detected. A human specialist will provide priority assistance."
            )

        # Rule 3: sensitive intent categories
        escalate_intents = {
            "device_hardware_issue": "Physical diagnostics required — specialist or Genius Bar visit recommended.",
            "account_billing": "Financial and account-security queries must be handled by a verified billing specialist.",
            "data_loss_recovery": "Potential data loss is a critical, time-sensitive issue requiring immediate specialist attention."
        }
        if intent in escalate_intents:
            return ("Escalate", escalate_intents[intent])

        # Rule 4: auto-handle
        auto_reasons = {
            "shipping_order": "Shipping status and policy questions are resolvable via self-service tracking tools.",
            "general_inquiry": "General informational queries can be resolved with published documentation and FAQs.",
            "connectivity_issue": "Standard connectivity troubleshooting steps are well-documented and effective for first-line resolution.",
            "software_bug": "Common software issues are addressed by documented troubleshooting steps and official update channels."
        }
        reason = auto_reasons.get(intent, "Standard inquiry resolvable with known policies and documentation.")
        return ("Auto-handle", reason)

    # ─────────────────────────────────────────────────────────────────────────
    def draft_reply(self, text, intent, confidence):
        """
        Draft a reply grounded in how the brand historically resolved similar issues.
        - For low confidence: returns a warm, generic escalation message.
        - Otherwise: uses TF-IDF cosine similarity to find the best-matching
          historical inbound query and returns its curated brand response.
        """
        if confidence < 0.38:
            return (
                "I want to make sure you get the best help possible! "
                "Let me connect you with one of our support specialists who can look into this in detail. "
                "You can also reach us directly at https://support.apple.com/contact — we're always happy to help! 🙏"
            )

        # Filter historical data for the predicted intent
        intent_subset = self.df[self.df['label_intent'] == intent].reset_index(drop=True)
        if len(intent_subset) == 0:
            intent_subset = self.df.reset_index(drop=True)

        # Cosine similarity against historical inbound messages
        subset_X = self.vectorizer.transform(intent_subset['user_text'])
        text_X = self.vectorizer.transform([text])
        similarities = cosine_similarity(text_X, subset_X).flatten()
        best_match_idx = int(np.argmax(similarities))

        historical_response = intent_subset.iloc[best_match_idx]['historical_brand_response']
        return historical_response

    # ─────────────────────────────────────────────────────────────────────────
    def get_similar_cases(self, text, intent, n=3):
        """Return top-N similar historical cases for transparency/explainability."""
        intent_subset = self.df[self.df['label_intent'] == intent].reset_index(drop=True)
        if len(intent_subset) < n:
            intent_subset = self.df.reset_index(drop=True)

        subset_X = self.vectorizer.transform(intent_subset['user_text'])
        text_X = self.vectorizer.transform([text])
        sims = cosine_similarity(text_X, subset_X).flatten()
        top_idx = np.argsort(sims)[::-1][:n]

        cases = []
        for idx in top_idx:
            cases.append({
                "query": intent_subset.iloc[idx]['user_text'],
                "response": intent_subset.iloc[idx]['historical_brand_response'],
                "similarity": round(float(sims[idx]), 3)
            })
        return cases


if __name__ == "__main__":
    agent = SupportAgent('data/AppleSupport_golden_labelled.csv')
    queries = [
        "My iPhone battery is draining incredibly fast on iOS 17",
        "I was charged twice for Apple Music",
        "Safari keeps crashing",
        "Where is my MacBook order?",
        "connect me to live support"
    ]
    for q in queries:
        intent, conf, top3 = agent.classify_intent(q)
        action, reason = agent.route_decision(q, intent, conf)
        draft = agent.draft_reply(q, intent, conf)
        print(f"\nQuery: {q}")
        print(f"  Intent: {intent} ({conf:.1%}) | Top3: {top3}")
        print(f"  Action: {action} — {reason}")
        print(f"  Draft:  {draft}")
