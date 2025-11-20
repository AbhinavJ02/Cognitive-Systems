# Agents Arguing Themselves into Hallucinations

A multi-agent LLM conversation system that explores how misinformation spreads or gets corrected in agent debates. Agents with distinct personalities (skeptic, optimist, persuader, deceiver) debate different types of claims to observe convergence toward accuracy or drift into misinformation.

## Documentation

- **[SETUP.md](SETUP.md)** - Complete setup and installation guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive code documentation explaining every part of the project
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide (5 minutes)

## Project Structure

- `agents/` - Agent personality definitions and configurations
- `data/` - Dataset of claims (ground truth, false, debatable)
- `judge/` - Judge agent for automated correctness classification
- `conversation/` - Core conversation system
- `evaluation/` - Parameter sweeps and evaluation metrics
- `visualization/` - Visualization tools for stance shifts and accuracy
- `logs/` - Conversation logs and results
- `config.py` - Configuration settings
- `main.py` - Main entry point

For detailed explanations of each file and component, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick Start

For detailed setup instructions, see [SETUP.md](SETUP.md).

### Quick Setup (3 steps):

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
```bash
# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

3. **Run the system:**
```bash
python example.py    # Simple example
python main.py       # Full system with menu
```

**For complete setup instructions, troubleshooting, and usage examples, see [SETUP.md](SETUP.md).**

## Features

- Multi-agent conversations with distinct personalities:
  - **Skeptic**: Challenges claims and asks for evidence
  - **Optimist**: Seeks consensus and tends to agree
  - **Persuader**: Strongly advocates for positions
  - **Deceiver**: Deliberately introduces false information to test misinformation spread
- Three claim types: ground truth facts, false statements, debatable claims
- Automated judge agent for correctness classification
- Parameter sweeps: context size, correctness, query complexity, turns, agents
- Visualization of stance shifts and accuracy metrics
- Comprehensive logging and tracking

