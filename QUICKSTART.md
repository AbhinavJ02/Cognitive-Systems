# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- OpenAI API key (get one at https://platform.openai.com/api-keys)

## Setup (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file with your API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 3. Run example
python example.py
```

That's it! If you see agents conversing, you're ready to go.

## Next Steps

- **Run full system**: `python main.py`
- **See detailed setup**: Read [SETUP.md](SETUP.md)
- **Understand the project**: Read [README.md](README.md)

## Troubleshooting

**"API key not found"** → Check your `.env` file format: `OPENAI_API_KEY=sk-...` (no quotes, no spaces)

**"Module not found"** → Run `pip install -r requirements.txt` again

**Need help?** → See [SETUP.md](SETUP.md) for detailed troubleshooting

