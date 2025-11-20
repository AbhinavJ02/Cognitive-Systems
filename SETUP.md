# Setup and Installation Guide

This guide will walk you through setting up and running the "Agents Arguing Themselves into Hallucinations" project on a new system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- An OpenAI API key (get one from https://platform.openai.com/api-keys)

## Step-by-Step Setup

### 1. Clone or Download the Project

If you have the project in a repository:
```bash
git clone <repository-url>
cd Cognitive
```

Or if you have the project files, navigate to the project directory:
```bash
cd /path/to/Cognitive
```

### 2. Create a Virtual Environment (Recommended)

Creating a virtual environment isolates the project dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when activated.

### 3. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- openai (OpenAI API client)
- python-dotenv (Environment variable management)
- matplotlib (Visualization)
- seaborn (Statistical visualization)
- pandas (Data manipulation)
- numpy (Numerical computing)
- tqdm (Progress bars)
- pydantic (Data validation)

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Or on Windows:
```cmd
type nul > .env
```

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important Notes:**
- Replace `sk-your-actual-api-key-here` with your actual OpenAI API key
- Do NOT include quotes around the key
- Do NOT include spaces around the `=` sign
- The format should be exactly: `OPENAI_API_KEY=sk-...`

Example:
```
OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE
```

### 5. Verify Setup

Test that everything is configured correctly:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
import config
from openai import OpenAI

# Test API connection
client = OpenAI(api_key=config.OPENAI_API_KEY)
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Say hello'}],
    max_tokens=5
)
print('✓ Setup successful! API connection works.')
print(f'✓ Response: {response.choices[0].message.content}')
"
```

If you see a success message, you're ready to go!

## Running the System

### Quick Start - Simple Example

Run a simple example conversation:

```bash
python example.py
```

This will:
- Load a debatable claim
- Run a conversation with 3 agents (skeptic, optimist, persuader)
- Evaluate the results with the judge agent
- Display the outcomes

### Full System - Interactive Menu

Run the main system with an interactive menu:

```bash
python main.py
```

You'll be prompted to choose a mode:
- `single` - Run a single conversation
- `sweep` - Run parameter sweeps (systematic evaluation)
- `baseline` - Run baseline comparison
- `all` - Run all modes

Or you can specify the mode directly:

```bash
python main.py single
python main.py sweep
python main.py baseline
python main.py all
```

### Understanding the Output

When you run a conversation, you'll see:

1. **Conversation Flow**: Each agent's responses as they debate the claim
2. **Judge Evaluation**: 
   - Overall correctness (correct/approximate/wrong)
   - Whether agents converged to the correct answer
   - Whether hallucinations were detected
3. **Results Files**: Saved in `logs/` directory as JSON files
4. **Visualizations**: Saved in `visualizations/` directory as PNG files

## Project Structure

```
Cognitive/
├── agents/              # Agent personality definitions
├── data/               # Dataset of claims (ground truth, false, debatable)
├── conversation/        # Core conversation system
├── judge/              # Judge agent for evaluation
├── evaluation/         # Parameter sweep system
├── visualization/      # Visualization tools
├── logs/               # Conversation logs (created automatically)
├── results/            # Parameter sweep results (created automatically)
├── visualizations/     # Generated plots (created automatically)
├── config.py           # Configuration settings
├── main.py             # Main entry point
├── example.py          # Simple example script
├── requirements.txt    # Python dependencies
└── .env                # Your API key (create this)
```

## Configuration Options

You can modify `config.py` to customize:

- **Model**: Change `DEFAULT_MODEL` (e.g., "gpt-4", "gpt-3.5-turbo", "gpt-4o-mini")
- **Temperature**: Adjust `DEFAULT_TEMPERATURE` (0.0-1.0, higher = more creative)
- **Max Turns**: Change `DEFAULT_MAX_TURNS` (number of conversation rounds)
- **Context Size**: Adjust `DEFAULT_CONTEXT_SIZE` (token limit)
- **Personalities**: Modify `AGENT_PERSONALITIES` to change agent behaviors

## Troubleshooting

### Issue: "OPENAI_API_KEY not found in environment variables"

**Solution:**
1. Make sure `.env` file exists in the project root
2. Check that the file contains exactly: `OPENAI_API_KEY=sk-...`
3. No quotes, no spaces around the `=` sign
4. Make sure you're running Python from the project directory

### Issue: "ModuleNotFoundError"

**Solution:**
1. Make sure you've activated your virtual environment
2. Run `pip install -r requirements.txt` again
3. Check that you're using Python 3.8+

### Issue: "API connection failed" or "Invalid API key"

**Solution:**
1. Verify your API key is correct in the `.env` file
2. Check that your OpenAI account has credits/quota
3. Make sure the API key starts with `sk-`
4. Try regenerating your API key on the OpenAI platform

### Issue: Import errors

**Solution:**
1. Make sure you're running scripts from the project root directory
2. Check that all `__init__.py` files exist in subdirectories
3. Try: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` (Linux/Mac) or `set PYTHONPATH=%CD%` (Windows)

### Issue: Permission errors

**Solution:**
1. Make sure you have write permissions in the project directory
2. The system will create `logs/`, `results/`, and `visualizations/` directories automatically

## Usage Examples

### Example 1: Run a single conversation

```bash
python main.py single
```

### Example 2: Run parameter sweeps

```bash
python main.py sweep
```

This will systematically test different:
- Context sizes
- Number of turns
- Personality combinations
- Claim types

Results are saved in `results/` and visualizations in `visualizations/`.

### Example 3: Compare baseline vs optimized

```bash
python main.py baseline
```

This compares default settings with optimized parameters.

### Example 4: Run everything

```bash
python main.py all
```

Runs all modes sequentially.

## Understanding the Results

### Log Files (`logs/`)

JSON files containing:
- Full conversation history
- Stance tracking for each agent
- Final responses
- Evaluation results
- Timing information

### Visualization Files (`visualizations/`)

PNG images showing:
- **Stance shifts**: How each agent's position changes over time
- **Accuracy vs parameters**: How different settings affect correctness
- **Convergence analysis**: Whether agents converge to correct answers
- **Hallucination detection**: When false information is introduced

### Results Files (`results/`)

JSON files from parameter sweeps containing:
- Parameter values tested
- Evaluation results for each configuration
- Best parameter recommendations

## Next Steps

1. **Explore the dataset**: Check `data/dataset.py` to see available claims
2. **Modify personalities**: Edit `config.py` to change agent behaviors
3. **Add your own claims**: Extend the dataset with your own test cases
4. **Experiment with parameters**: Try different models, temperatures, and turn counts
5. **Analyze results**: Review logs and visualizations to understand agent behavior

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify your `.env` file format
3. Make sure all dependencies are installed
4. Check that you're using Python 3.8+
5. Verify your OpenAI API key is valid and has credits

## Security Note

⚠️ **Important**: Never commit your `.env` file to version control. The `.gitignore` file is already configured to exclude it, but always double-check before pushing to a repository.

---

**You're all set!** Start with `python example.py` to see the system in action.

