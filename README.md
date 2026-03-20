# Call Match Package Documentation

> Call Quality Analysis System - An automated tool for evaluating SDR (Sales Development Representative) call quality

---

## Overview

This package contains two core features:

| Step | Script | Function |
|------|--------|----------|
| **Step 1** | `csv_to_call_opening.py` | Import opening templates from CSV to database |
| **Step 2** | `analyze_calls.py` | Analyze call records and calculate three matching scores |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with actual values
```

### 3. Run Scripts

```bash
# Step 1: Import openings
python csv_to_call_opening.py

# Step 2: Analyze calls
python analyze_calls.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DB_HOST` | Yes | localhost | Database host |
| `DB_PORT` | No | 3306 | Database port |
| `DB_USER` | Yes | root | Database username |
| `DB_PASSWORD` | Yes | - | Database password |
| `DB_NAME` | Yes | sdr_backend | Database name |
| `LLM_API_KEY` | Yes | - | OpenAI API Key |
| `LLM_BASE_URL` | No | https://api.openai.com/v1 | API endpoint |
| `LLM_MODEL` | No | gpt-4o-mini | Model name |
| `CSV_FILE` | No | ./openings.csv | CSV file path |

---

## Business Configuration

Edit the following content in `config.py` to adapt to your business:

### Preset Questions and Answers

```python
QUESTIONS_ANSWERS = [
    {
        "question": "Your first question?",
        "answers": ["Answer 1", "Answer 2", "Answer 3"],
    },
]
```

### Script Template

```python
FOLLOW_UP_SCRIPT = """
Your SDR script content...
"""
```

---

## Data Flow

```
1. Read company openings from CSV
   ↓
2. Write to database call_opening field
   ↓
3. Analyze call records
   ↓
4. Calculate three matching scores
   ↓
5. Write back to database
```

---

## Output Fields

| Field | Description | Evaluated Object |
|-------|-------------|------------------|
| `opening_match` | Opening match score (0-100) | Dialer |
| `question_match` | Question usage rate (0-100) | Dialer |
| `answer_match` | Answer match score (0-100) | Receiver |

---

## Notes

1. All sensitive information (database passwords, API keys) must be configured through environment variables
2. Please modify script templates and question-answer pairs according to actual business requirements
3. Ensure database table structure is correct before first use

---

## License

Apache-2.0 License
