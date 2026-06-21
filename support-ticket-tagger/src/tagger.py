"""
Support Ticket Auto-Tagger
Implements zero-shot, few-shot, and comparison logic using OpenRouter API.
"""

import json
import os
import time
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# OpenRouter config
# ──────────────────────────────────────────────
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"

# ──────────────────────────────────────────────
# Available tag taxonomy
# ──────────────────────────────────────────────
ALL_TAGS = [
    "Bug Report", "Feature Request", "Billing", "Account Access",
    "Performance", "API Issue", "UI/UX", "How-To", "Documentation",
    "Data Recovery", "Compliance", "Integration", "Mobile App",
    "Notifications", "Security", "Crash", "Refund Request",
    "Account Upgrade", "Account Downgrade", "Data Export",
    "File Upload", "Search", "Developer", "Urgent", "2FA",
    "Password Reset", "Email Issue", "Dark Mode", "Dashboard",
    "GDPR", "Localization", "Webhook", "Enterprise", "Non-Profit",
    "Duplicate Charge", "SMS Issue", "Update Issue", "Discount Request",
    "Data Issue", "Data Deletion", "Account Locked", "Account Issue",
    "Size Limit", "Internationalization"
]

# ──────────────────────────────────────────────
# Few-shot examples
# ──────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "ticket": "I can't log in to my account. The password reset email never comes.",
        "tags": ["Account Access", "Password Reset", "Email Issue"]
    },
    {
        "ticket": "Please cancel my subscription and refund my last payment.",
        "tags": ["Billing", "Refund Request", "Account Downgrade"]
    },
    {
        "ticket": "The mobile app crashes immediately on Android 13 after the latest update.",
        "tags": ["Bug Report", "Mobile App", "Crash"]
    },
    {
        "ticket": "How do I connect your service with Zapier for automation?",
        "tags": ["How-To", "Integration", "Feature Request"]
    },
    {
        "ticket": "The API rate limit docs are wrong — limits are different in production.",
        "tags": ["Documentation", "API Issue", "Developer"]
    }
]


class SupportTicketTagger:
    """Tags support tickets using OpenRouter LLM with zero-shot and few-shot strategies."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")
        self.model = MODEL

    def _call_api(self, prompt: str) -> tuple:
        """Call OpenRouter API and return (response_text, latency_seconds)."""
        start = time.time()
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://support-ticket-tagger",
                "X-Title": "Support Ticket Auto-Tagger"
            },
            json={
                "model": self.model,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        elapsed = round(time.time() - start, 2)

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")

        text = response.json()["choices"][0]["message"]["content"].strip()
        return text, elapsed

    def _parse_response(self, raw: str) -> Dict:
        """Strip markdown fences and parse JSON."""
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    def zero_shot_tag(self, ticket_text: str) -> Dict:
        """Tag a ticket with no examples — pure instruction prompting."""
        prompt = f"""You are an expert support ticket classification system.

Your task is to analyze the support ticket below and assign the TOP 3 most relevant tags from the provided taxonomy.

AVAILABLE TAGS:
{', '.join(ALL_TAGS)}

INSTRUCTIONS:
- Select exactly 3 tags ranked by relevance (most relevant first)
- Choose tags that best capture the ticket's main issue, secondary concern, and category
- Output ONLY a valid JSON object — no explanation, no extra text

OUTPUT FORMAT:
{{
  "tags": ["Tag1", "Tag2", "Tag3"],
  "confidence": [0.95, 0.80, 0.65],
  "reasoning": "One sentence explaining your choices"
}}

SUPPORT TICKET:
\"\"\"{ticket_text}\"\"\"
"""
        raw, elapsed = self._call_api(prompt)
        result = self._parse_response(raw)
        result["strategy"] = "zero-shot"
        result["latency_s"] = elapsed
        return result

    def few_shot_tag(self, ticket_text: str) -> Dict:
        """Tag a ticket using few-shot examples to guide the model."""
        examples_block = ""
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            examples_block += f"""
Example {i}:
Ticket: \"{ex['ticket']}\"
Output: {json.dumps({"tags": ex["tags"], "confidence": [0.95, 0.82, 0.70], "reasoning": "Primary issue identified with supporting context."})}
"""

        prompt = f"""You are an expert support ticket classification system.

Your task is to analyze support tickets and assign the TOP 3 most relevant tags.

AVAILABLE TAGS:
{', '.join(ALL_TAGS)}

INSTRUCTIONS:
- Select exactly 3 tags ranked by relevance (most relevant first)
- Learn the pattern from the examples below
- Output ONLY a valid JSON object — no explanation, no extra text

OUTPUT FORMAT:
{{
  "tags": ["Tag1", "Tag2", "Tag3"],
  "confidence": [0.95, 0.80, 0.65],
  "reasoning": "One sentence explaining your choices"
}}

--- FEW-SHOT EXAMPLES ---
{examples_block}
-------------------------

Now classify this ticket:
Ticket: \"\"\"{ticket_text}\"\"\"
Output:"""

        raw, elapsed = self._call_api(prompt)
        result = self._parse_response(raw)
        result["strategy"] = "few-shot"
        result["latency_s"] = elapsed
        return result

    def compare(self, ticket_text: str, true_tags: List[str] = None) -> Dict:
        """Run both strategies and return a comparison."""
        zero = self.zero_shot_tag(ticket_text)
        few = self.few_shot_tag(ticket_text)

        def top1_match(pred, truth):
            return pred["tags"][0] in truth if truth else None

        def top3_match(pred, truth):
            return len(set(pred["tags"]) & set(truth)) if truth else None

        return {
            "ticket": ticket_text,
            "true_tags": true_tags,
            "zero_shot": zero,
            "few_shot": few,
            "evaluation": {
                "zero_shot_top1_correct": top1_match(zero, true_tags),
                "few_shot_top1_correct": top1_match(few, true_tags),
                "zero_shot_top3_overlap": top3_match(zero, true_tags),
                "few_shot_top3_overlap": top3_match(few, true_tags),
            } if true_tags else None
        }

    def evaluate_dataset(self, tickets: List[Dict], max_tickets: int = 10) -> Dict:
        """Evaluate both strategies across a dataset subset."""
        results = []
        sample = tickets[:max_tickets]

        print(f"\n🔍 Evaluating {len(sample)} tickets...\n")
        for i, ticket in enumerate(sample):
            print(f"  Processing TKT {i+1}/{len(sample)}: {ticket['id']}")
            comparison = self.compare(ticket["text"], ticket.get("true_tags"))
            comparison["id"] = ticket["id"]
            results.append(comparison)
            time.sleep(0.3)

        zero_top1 = [r["evaluation"]["zero_shot_top1_correct"] for r in results if r["evaluation"]]
        few_top1  = [r["evaluation"]["few_shot_top1_correct"]  for r in results if r["evaluation"]]
        zero_top3 = [r["evaluation"]["zero_shot_top3_overlap"] for r in results if r["evaluation"]]
        few_top3  = [r["evaluation"]["few_shot_top3_overlap"]  for r in results if r["evaluation"]]

        summary = {
            "total_tickets": len(results),
            "zero_shot": {
                "top1_accuracy": round(sum(zero_top1) / len(zero_top1), 3) if zero_top1 else None,
                "avg_top3_overlap": round(sum(zero_top3) / len(zero_top3), 3) if zero_top3 else None,
                "avg_latency_s": round(sum(r["zero_shot"]["latency_s"] for r in results) / len(results), 2)
            },
            "few_shot": {
                "top1_accuracy": round(sum(few_top1) / len(few_top1), 3) if few_top1 else None,
                "avg_top3_overlap": round(sum(few_top3) / len(few_top3), 3) if few_top3 else None,
                "avg_latency_s": round(sum(r["few_shot"]["latency_s"] for r in results) / len(results), 2)
            }
        }

        return {"results": results, "summary": summary}