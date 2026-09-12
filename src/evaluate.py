import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score
from agent import SupportAgent
import difflib

# Attempt to load OpenAI for LLM-as-judge
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

def exact_match_ratio(str1, str2):
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

class Evaluator:
    def __init__(self, golden_path, agent):
        self.df = pd.read_csv(golden_path)
        self.agent = agent
        
        # Check for API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key and HAS_OPENAI:
            self.client = OpenAI(api_key=self.api_key)
            self.use_llm_judge = True
        else:
            self.use_llm_judge = False
            print("WARNING: OPENAI_API_KEY not found or openai package missing.")
            print("Falling back to heuristic text similarity for Response Quality (LLM-as-Judge simulated).")

    def evaluate_reply_quality(self, user_text, drafted_reply, expected_reply):
        """
        Uses an LLM-as-judge to score the reply from 1 to 5 based on:
        - Relevance to user text
        - Alignment with brand historical expected reply
        - Professionalism
        """
        if self.use_llm_judge:
            prompt = f"""
            You are an expert customer support QA manager. Evaluate the AI-drafted reply based on the user's issue and how the brand historically responds.
            
            User Issue: {user_text}
            Expected Brand Reply (Historical): {expected_reply}
            AI Drafted Reply: {drafted_reply}
            
            Score the AI Drafted Reply on a scale of 1 to 5, where:
            1 = Completely irrelevant or harmful
            3 = Acceptable, but misses nuances of the expected response
            5 = Perfect alignment with brand guidelines and fully resolves the issue
            
            Output ONLY the integer score (1, 2, 3, 4, or 5).
            """
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0
                )
                score = int(response.choices[0].message.content.strip())
                return score
            except Exception as e:
                return 3 # Neutral fallback on API error
        else:
            # Fallback heuristic: 5 if highly similar, 3 if somewhat similar, 1 if completely different
            similarity = exact_match_ratio(drafted_reply, expected_reply)
            if similarity > 0.8: return 5
            elif similarity > 0.4: return 3
            else: return 1

    def run_evaluation(self):
        results = {
            "intent_true": [],
            "intent_pred": [],
            "action_true": [],
            "action_pred": [],
            "reply_scores": [],
            "failures": []
        }
        
        # Baselines
        trivial_action_pred = [] # Trivial: Always escalate
        
        print(f"Evaluating {len(self.df)} examples...")
        
        for idx, row in self.df.iterrows():
            # True labels
            true_intent = row['label_intent']
            true_action = row['label_action']
            expected_reply = row['historical_brand_response']
            user_text = row['user_text']
            
            # Agent Prediction
            pred_intent, conf, _top3 = self.agent.classify_intent(user_text)
            pred_action, reason = self.agent.route_decision(user_text, pred_intent, conf)
            draft = self.agent.draft_reply(user_text, pred_intent, conf)
            
            # Judge
            score = self.evaluate_reply_quality(user_text, draft, expected_reply)
            
            results["intent_true"].append(true_intent)
            results["intent_pred"].append(pred_intent)
            results["action_true"].append(true_action)
            results["action_pred"].append(pred_action)
            results["reply_scores"].append(score)
            
            trivial_action_pred.append("Escalate")
            
            # Log failures for analysis (Intent mismatch or Routing mismatch)
            if pred_intent != true_intent or pred_action != true_action:
                if len(results["failures"]) < 5: # Keep top 5 failures
                    results["failures"].append({
                        "user_text": user_text,
                        "true_intent": true_intent,
                        "pred_intent": pred_intent,
                        "true_action": true_action,
                        "pred_action": pred_action,
                        "draft": draft,
                        "reason": reason
                    })
                    
        # Compute Metrics
        intent_acc = accuracy_score(results["intent_true"], results["intent_pred"])
        action_acc = accuracy_score(results["action_true"], results["action_pred"])
        trivial_action_acc = accuracy_score(results["action_true"], trivial_action_pred)
        avg_reply_score = np.mean(results["reply_scores"])
        
        metrics = {
            "Intent_Accuracy": round(intent_acc, 3),
            "Routing_Accuracy": round(action_acc, 3),
            "Baseline_Trivial_Routing_Accuracy": round(trivial_action_acc, 3),
            "Avg_LLM_Judge_Reply_Score_out_of_5": round(avg_reply_score, 2),
            "Total_Evaluated": len(self.df)
        }
        
        print("\n=== EVALUATION RESULTS ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
        with open("metrics.json", "w") as f:
            json.dump({"metrics": metrics, "top_failures": results["failures"]}, f, indent=4)
        print("\nDetailed metrics and top 5 failures saved to metrics.json")

if __name__ == "__main__":
    golden_file = 'data/AppleSupport_golden_labelled.csv'
    if not os.path.exists(golden_file):
        print(f"Error: {golden_file} not found. Please run data generation script first.")
        exit(1)
        
    print("Initializing Agent...")
    agent = SupportAgent(golden_file)
    
    print("Starting Evaluation Harness...")
    evaluator = Evaluator(golden_file, agent)
    evaluator.run_evaluation()
