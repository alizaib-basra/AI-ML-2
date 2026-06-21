# 🏷️ Support Ticket Auto-Tagger

Automatically classify support tickets into ranked tags using LLM prompt engineering — comparing **Zero-Shot** vs **Few-Shot** strategies with the Claude API.

---

## ✨ Features

- 🔹 **Zero-Shot Classification** — Pure instruction prompting, no examples
- 🔸 **Few-Shot Classification** — Guided by curated examples for higher accuracy
- 📊 **Side-by-Side Comparison** — Evaluate both strategies on the same ticket
- 🏆 **Top 3 Tags with Confidence Scores** — Ranked multi-label output
- 🌐 **Interactive Web Demo** — Browser-based UI with preset tickets
- 📦 **Batch Evaluation** — Run over a labeled dataset and compare accuracy metrics
- 🏷️ **44-Tag Taxonomy** — Covers billing, bugs, access, compliance, integrations, and more

---

## 📁 Project Structure

```
support-ticket-tagger/
├── data/
│   └── tickets.json          # 20 labeled support tickets dataset
├── src/
│   └── tagger.py             # Core LLM tagger (zero-shot, few-shot, batch eval)
├── results/                  # Auto-created; batch output saved here
├── main.py                   # CLI runner
├── demo.html                 # Interactive web demo (no server needed)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the CLI

**Tag a single ticket (compare both strategies):**
```bash
python main.py --mode compare --ticket "My app crashes every time I open it on iPhone 14"
```

**Zero-shot only:**
```bash
python main.py --mode single --strategy zero --ticket "I was charged twice this month"
```

**Few-shot only:**
```bash
python main.py --mode single --strategy few --ticket "How do I export my data to CSV?"
```

**Batch evaluate the dataset:**
```bash
python main.py --mode batch --max 10 --output results/eval.json
```

---

## 🌐 Web Demo

Open `demo.html` directly in your browser — no server needed.

1. Enter your Anthropic API key in the input
2. Choose a strategy: Zero-Shot, Few-Shot, or Compare Both
3. Type or pick a preset support ticket
4. Click **▶ Tag Ticket** (or press `Ctrl+Enter`)

---

## 🧠 Prompt Engineering Strategies

### Zero-Shot
The model receives only the tag taxonomy and instructions — no labeled examples. Tests the model's inherent understanding of support ticket categories.

```
You are an expert support ticket classification system.
Assign the TOP 3 most relevant tags from this taxonomy: [...]
Output JSON: {"tags": [...], "confidence": [...], "reasoning": "..."}
```

### Few-Shot
Five curated examples are prepended to the prompt, showing the model the expected input→output pattern. Improves accuracy on ambiguous or edge-case tickets.

```
Example 1:
Ticket: "I can't log in. Reset email never arrives."
Output: {"tags": ["Account Access", "Password Reset", "Email Issue"], ...}
...
Now classify: [new ticket]
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | % of tickets where the #1 predicted tag is in the ground truth |
| **Avg Top-3 Overlap** | Average number of predicted tags (out of 3) matching ground truth |
| **Avg Latency (s)** | Average API response time per ticket |

---

## 🏷️ Tag Taxonomy (44 tags)

```
Bug Report, Feature Request, Billing, Account Access, Performance,
API Issue, UI/UX, How-To, Documentation, Data Recovery, Compliance,
Integration, Mobile App, Notifications, Security, Crash, Refund Request,
Account Upgrade, Account Downgrade, Data Export, File Upload, Search,
Developer, Urgent, 2FA, Password Reset, Email Issue, Dark Mode, Dashboard,
GDPR, Localization, Webhook, Enterprise, Non-Profit, Duplicate Charge,
SMS Issue, Update Issue, Discount Request, Data Issue, Data Deletion,
Account Locked, Account Issue, Size Limit, Internationalization
```

---

## 📋 Dataset

20 hand-labeled support tickets covering diverse real-world scenarios:
- Account & access issues
- Billing and payment problems
- Technical bugs and crashes
- Feature and integration requests
- Compliance and legal (GDPR)
- Developer and API issues

---

## 🔬 Key Findings

- **Few-Shot outperforms Zero-Shot** on ambiguous tickets where category boundaries overlap
- **Zero-Shot performs well** on clear-cut tickets (billing, password reset, obvious crashes)
- **Confidence calibration** is generally good — high confidence correlates with correct tags
- **Latency** is slightly higher for Few-Shot due to longer prompts (~10–15% overhead)

---

## 🛠️ Skills Demonstrated

- ✅ Prompt Engineering (system prompts, output constraints, JSON formatting)
- ✅ Zero-Shot Learning
- ✅ Few-Shot Learning with curated examples
- ✅ LLM-based multi-label text classification
- ✅ Confidence-ranked multi-class prediction
- ✅ Batch evaluation and metric comparison

---

## 📄 License

MIT — free to use, modify, and build upon.
