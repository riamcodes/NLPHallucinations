# Simple Question Answering with DistilBERT and GPT-2



## What You Need

- Python 3.8+
- 2 packages: `torch` and `transformers`

## Quick Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install torch transformers

# Run the system
python main.py
```

## Every Time You Want to Run It

```bash
# Navigate to project directory
cd /path/to/NLP5

# Activate virtual environment
source venv/bin/activate

# Run the system
python main.py
```

## Yiippie yippie yippie hopeefully it worked

The system will:
1. Load DistilBERT and GPT-2 models
2. Answer 2 sample questions
3. Generate additional text with GPT-2
4. Show you the results

## How It Works

- **1 file**: `main.py` (163 lines)
- **2 models**: DistilBERT for Q&A + GPT-2 for text generation
- **Simple**: Just load and ask questions

## Example Output

```
Question 1: What is the capital of France?
Answer: Paris
Confidence: 0.987

Generated Text:
Question: What is the capital of France?
Context: France is a country in Western Europe...
Answer: Paris
Additional information: France is known for its rich history...
Generated Length: 25 words

Question 2: Who wrote Romeo and Juliet?
Answer: William Shakespeare
Confidence: 0.923

Generated Text:
Question: Who wrote Romeo and Juliet?
Context: Romeo and Juliet is a tragedy...
Answer: William Shakespeare
Additional information: Shakespeare was an English playwright...
Generated Length: 22 words
```

## For Your Own Questions

Edit the `questions` list in `main.py`:

```python
questions = [
    {
        "question": "Your question here?",
        "context": "Your context here."
    }
]
```

## Troubleshooting

### "externally-managed-environment" Error
If you get this error when running `pip install`:
```bash
error: externally-managed-environment
```

**Solution:** Use the virtual environment setup above. This error happens when trying to install packages globally on macOS.

### Virtual Environment Not Working
If `source venv/bin/activate` doesn't work:
```bash
# Make sure you're in the right directory
cd /path/to/NLP5

# Check if venv folder exists
ls -la

# If venv doesn't exist, create it
python3 -m venv venv
source venv/bin/activate
```

### Models Not Downloading
If models fail to download:
- Check your internet connection
- Ensure you have enough disk space (1GB+)
- Try running the command again

**That's it! No complex setup, no multiple files, just simple Q&A with text generation.**