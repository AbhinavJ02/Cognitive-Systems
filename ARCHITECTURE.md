# Project Architecture and Code Documentation

This document provides a comprehensive explanation of every part of the project, including all code files, their contents, and how they work together.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Core Configuration](#core-configuration)
5. [Data Layer](#data-layer)
6. [Agent System](#agent-system)
7. [Conversation System](#conversation-system)
8. [Judge System](#judge-system)
9. [Evaluation System](#evaluation-system)
10. [Visualization System](#visualization-system)
11. [Entry Points](#entry-points)
12. [Data Flow](#data-flow)
13. [Key Classes and Functions](#key-classes-and-functions)

---

## Project Overview

This project implements a multi-agent LLM conversation system where agents with distinct personalities debate claims to observe whether they converge toward accuracy or drift into misinformation/hallucinations.

**Core Concept**: Multiple AI agents (skeptic, optimist, persuader, deceiver) engage in conversations about different types of claims (ground truth facts, false statements, debatable topics). A judge agent evaluates the outcomes to determine correctness, convergence, and hallucination detection.

---

## Architecture Overview

The project follows a modular architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Points                         │
│  (main.py, example.py)                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│  Conversation  │   │   Judge Agent    │
│   Manager      │   │                  │
└───────┬────────┘   └────────┬─────────┘
        │                     │
        │         ┌───────────┴──────────┐
        │         │                      │
┌───────▼─────────▼──────┐   ┌───────────▼──────────┐
│   Agent Personalities │   │   Dataset (Claims)   │
│   (config.py)         │   │   (data/dataset.py)   │
└───────────────────────┘   └──────────────────────┘
        │
        │
┌───────▼──────────────────────────────────────────┐
│         OpenAI API (External Service)            │
└──────────────────────────────────────────────────┘
```

---

## Directory Structure

```
Cognitive/
├── agents/                    # Agent personality definitions
│   ├── __init__.py           # Package initialization
│   └── agent_factory.py      # Personality configuration utilities
│
├── data/                      # Dataset of claims
│   ├── __init__.py           # Package initialization
│   └── dataset.py            # Claim definitions and dataset
│
├── conversation/              # Core conversation system
│   ├── __init__.py           # Package initialization
│   └── conversation_manager.py  # Manages multi-agent conversations
│
├── judge/                     # Judge agent for evaluation
│   ├── __init__.py           # Package initialization
│   └── judge_agent.py        # Automated correctness classification
│
├── evaluation/               # Parameter sweep system
│   ├── __init__.py           # Package initialization
│   └── parameter_sweep.py    # Systematic parameter testing
│
├── visualization/            # Visualization tools
│   ├── __init__.py           # Package initialization
│   └── visualizer.py         # Plotting and visualization
│
├── logs/                      # Conversation logs (auto-created)
├── results/                   # Parameter sweep results (auto-created)
├── visualizations/            # Generated plots (auto-created)
│
├── config.py                  # Global configuration
├── main.py                    # Main entry point
├── example.py                 # Simple example script
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API key)
└── .gitignore                 # Git ignore rules
```

---

## Core Configuration

### `config.py`

**Purpose**: Central configuration file containing all system settings, agent personalities, and default parameters.

**Key Components**:

1. **API Configuration** (lines 9-12)
   - Loads `OPENAI_API_KEY` from environment variables
   - Raises error if key is missing
   - Uses `python-dotenv` to load from `.env` file

2. **Model Configuration** (lines 14-16)
   - `DEFAULT_MODEL`: OpenAI model to use (default: "gpt-4o-mini")
   - `DEFAULT_TEMPERATURE`: Model creativity/randomness (default: 0.7)

3. **Agent Personalities** (lines 18-44)
   - Dictionary defining four agent types:
     - **Skeptic**: Challenges claims, asks for evidence, analytical
     - **Optimist**: Seeks consensus, tends to agree, positive
     - **Persuader**: Strongly advocates, uses rhetoric, assertive
     - **Deceiver**: Deliberately introduces false information, misleads others, manipulative
   - Each personality has:
     - `role`: Agent's role name
     - `goal`: What the agent aims to do
     - `backstory`: Detailed personality description
     - `personality_traits`: List of trait keywords

4. **Conversation Parameters** (lines 40-43)
   - `DEFAULT_CONTEXT_SIZE`: Token limit (default: 2000)
   - `DEFAULT_MAX_TURNS`: Maximum conversation rounds (default: 10)
   - `DEFAULT_NUM_AGENTS`: Number of agents (default: 3)

5. **Evaluation Settings** (lines 48-50)
   - `JUDGE_MODEL`: Model for judge agent (default: "gpt-4o-mini")
   - `JUDGE_TEMPERATURE`: Lower temperature for consistent judgments (default: 0.3)

6. **Directory Settings** (lines 52-54)
   - `LOG_DIR`: Where conversation logs are saved
   - `RESULTS_DIR`: Where parameter sweep results are saved

**Usage**: Imported by all modules to access configuration settings.

---

## Data Layer

### `data/dataset.py`

**Purpose**: Defines the dataset of claims used for testing conversations.

**Key Components**:

1. **ClaimType Enum** (lines 7-10)
   - `GROUND_TRUTH`: Factually correct statements
   - `FALSE`: Factually incorrect statements
   - `DEBATABLE`: Statements with valid arguments on both sides

2. **Claim Class** (lines 12-19)
   - Represents a single claim to be debated
   - Attributes:
     - `text`: The claim statement
     - `claim_type`: Type of claim (ClaimType enum)
     - `ground_truth`: Correct answer (defaults to text for ground truth claims)

3. **DATASET Dictionary** (lines 22-53)
   - Contains 8 claims per category (24 total):
     - **Ground Truth**: Scientific facts, established knowledge
     - **False**: Common misconceptions, false beliefs
     - **Debatable**: Controversial topics, opinions
   - Organized by `ClaimType` as keys

4. **Utility Functions**:
   - `get_claims_by_type(claim_type)`: Returns all claims of a specific type
   - `get_all_claims()`: Returns all claims from all categories
   - `get_claim_statistics()`: Returns count of claims per category

**Example Usage**:
```python
from data.dataset import ClaimType, get_claims_by_type
claims = get_claims_by_type(ClaimType.DEBATABLE)
```

---

## Agent System

### `agents/agent_factory.py`

**Purpose**: Provides utilities for accessing agent personality configurations.

**Key Functions**:

1. **`get_personality_config(personality: str) -> Dict`** (lines 9-22)
   - Retrieves configuration for a specific personality
   - Validates personality name
   - Returns dictionary with role, goal, backstory, traits

2. **`get_available_personalities() -> List[str]`** (lines 24-26)
   - Returns list of all available personality types
   - Currently: ['skeptic', 'optimist', 'persuader', 'deceiver']

**Note**: This module doesn't create actual agent instances. Agent interactions are handled directly via OpenAI API in `conversation_manager.py`. This module provides personality configurations that are used to construct prompts.

---

## Conversation System

### `conversation/conversation_manager.py`

**Purpose**: Core system that manages multi-agent conversations, tracks stance changes, and coordinates agent interactions.

**Key Class**: `ConversationManager`

#### Initialization (`__init__`, lines 13-42)

**Parameters**:
- `personalities`: List of agent personalities (default: all three)
- `model`: OpenAI model to use (default: from config)
- `temperature`: Model temperature (default: from config)
- `max_turns`: Maximum conversation rounds (default: 10)
- `context_size`: Token limit (default: 2000)

**State Variables**:
- `conversation_history`: List of all messages in the conversation
- `stance_tracking`: Dictionary tracking each agent's stance over time
- `current_turn`: Current turn number
- `start_time` / `end_time`: Timing information

#### Key Methods:

1. **`_get_agent_prompt(personality, claim, conversation_so_far)`** (lines 42-68)
   - Generates a prompt for an agent based on their personality
   - Includes:
     - Personality traits and backstory
     - The claim being discussed
     - Previous conversation context (last 5 messages)
   - Returns formatted prompt string

2. **`_call_agent(personality, claim, conversation_so_far)`** (lines 70-88)
   - Makes API call to OpenAI for a specific agent
   - Uses agent's backstory as system message
   - Uses generated prompt as user message
   - Returns agent's response text
   - Handles errors gracefully

3. **`_extract_stance(message, personality)`** (lines 90-106)
   - Extracts stance information from agent's message
   - Uses simple keyword-based sentiment analysis:
     - "agree", "correct", "true" → positive sentiment
     - "disagree", "wrong", "false" → negative sentiment
     - Otherwise → neutral
   - Returns dictionary with turn, message, timestamp, sentiment

4. **`run_conversation(claim)`** (lines 108-198)
   - Main method that orchestrates the conversation
   - Process:
     1. Initialize conversation state
     2. First round: Each agent gives initial stance
     3. Subsequent rounds: Agents respond to each other
     4. Track all messages and stance changes
     5. Compile results
   - Returns dictionary with:
     - Claim and claim type
     - Full conversation history
     - Stance tracking for each agent
     - Final responses from each agent
     - Number of turns, duration, configuration

**Data Flow**:
```
Claim → ConversationManager → OpenAI API → Agent Responses → Stance Tracking → Results
```

---

## Judge System

### `judge/judge_agent.py`

**Purpose**: Evaluates conversations to determine correctness, convergence, and hallucination detection.

**Key Components**:

1. **CorrectnessLevel Enum** (lines 9-12)
   - `CORRECT`: Agent's position is correct
   - `APPROXIMATE`: Agent's position is partially correct
   - `WRONG`: Agent's position is incorrect

2. **JudgeAgent Class** (lines 14-153)

#### Initialization (`__init__`, lines 15-25)
- Uses lower temperature (0.3) for more consistent judgments
- Initializes OpenAI client

#### Key Methods:

1. **`evaluate_conversation(claim, claim_type, final_responses, conversation_history)`** (lines 27-100)
   - Main evaluation method
   - Process:
     1. Builds evaluation prompt with:
        - Original claim and type
        - Expected evaluation based on claim type
        - Final responses from all agents
     2. Calls OpenAI API with judge prompt
     3. Parses evaluation response
     4. Returns structured evaluation dictionary
   - Returns dictionary with:
     - `overall_correctness`: correct/approximate/wrong
     - `agent_evaluations`: Per-agent correctness
     - `converged`: Whether agents converged to correct answer
     - `hallucinations_detected`: Whether false facts were introduced
     - `reasoning`: Judge's explanation
     - `raw_evaluation`: Full judge response

2. **`_parse_evaluation(evaluation_text, final_responses)`** (lines 102-153)
   - Parses judge's text response into structured format
   - Extracts:
     - Overall correctness from text
     - Individual agent evaluations
     - Convergence status
     - Hallucination detection
   - Uses keyword matching (could be enhanced with more sophisticated parsing)

3. **`classify_correctness(correctness_str)`** (lines 155-163)
   - Converts correctness string to `CorrectnessLevel` enum
   - Helper method for type conversion

**Evaluation Logic**:
- For `ground_truth` claims: Expects CORRECT evaluation
- For `false` claims: Expects WRONG evaluation
- For `debatable` claims: Accepts valid arguments on both sides

---

## Evaluation System

### `evaluation/parameter_sweep.py`

**Purpose**: Systematically tests different parameter combinations to find optimal settings.

**Key Class**: `ParameterSweep`

#### Initialization (`__init__`, lines 13-21)
- Creates results directory
- Initializes results list

#### Key Methods:

1. **`sweep_context_size(context_sizes, claim, ...)`** (lines 23-58)
   - Tests different context size values
   - For each context size:
     - Creates ConversationManager with that context size
     - Runs conversation
     - Evaluates with JudgeAgent
     - Records results
   - Returns list of results

2. **`sweep_max_turns(max_turns_list, claim, ...)`** (lines 60-95)
   - Tests different numbers of conversation turns
   - Similar process to context size sweep

3. **`sweep_personality_combinations(personality_combinations, claim, ...)`** (lines 97-132)
   - Tests different combinations of agent personalities
   - Examples: [skeptic, optimist], [skeptic, persuader], [skeptic, optimist, deceiver], all four, etc.

4. **`sweep_claim_types(claims, ...)`** (lines 134-169)
   - Tests different claim types
   - Runs conversations for multiple claims

5. **`save_results(filename)`** (lines 171-185)
   - Saves all results to JSON file
   - Timestamped filename by default
   - Saves to `results/` directory

6. **`get_best_parameters()`** (lines 187-225)
   - Analyzes all results to find best parameter values
   - Calculates correctness rates for each parameter value
   - Returns dictionary with best values and their scores

**Result Structure**:
Each result contains:
- Parameter name and value tested
- Claim information
- Full conversation result
- Judge evaluation
- Timestamp

---

## Visualization System

### `visualization/visualizer.py`

**Purpose**: Creates visualizations of conversation results, stance shifts, and parameter sweeps.

**Key Class**: `Visualizer`

#### Initialization (`__init__`, lines 13-20)
- Creates output directory for visualizations
- Sets matplotlib/seaborn style

#### Key Methods:

1. **`plot_stance_shifts(conversation_result, save_path)`** (lines 22-60)
   - Plots how each agent's stance changes over conversation turns
   - X-axis: Turn number
   - Y-axis: Stance sentiment (-1 negative, 0 neutral, +1 positive)
   - Different line for each agent personality
   - Saves as PNG file

2. **`plot_accuracy_vs_parameter(sweep_results, parameter_name, save_path)`** (lines 62-110)
   - Bar chart showing accuracy for different parameter values
   - Groups by correctness level (correct/approximate/wrong)
   - X-axis: Parameter values
   - Y-axis: Count of each correctness level
   - Useful for parameter sweep analysis

3. **`plot_convergence_analysis(sweep_results, save_path)`** (lines 112-150)
   - Heatmap showing convergence rates
   - Rows: Different parameters
   - Columns: Parameter values
   - Color intensity: Convergence rate (0-1)
   - Helps identify which parameter combinations lead to convergence

4. **`plot_hallucination_detection(sweep_results, save_path)`** (lines 152-200)
   - Bar chart showing hallucination rates
   - Groups by parameter and claim type
   - Shows when false information is introduced
   - Helps identify problematic configurations

**Visualization Output**:
- All plots saved as PNG files in `visualizations/` directory
- High resolution (300 DPI)
- Professional styling with seaborn

---

## Entry Points

### `main.py`

**Purpose**: Main entry point with interactive menu and multiple execution modes.

**Key Functions**:

1. **`run_single_conversation(claim, personalities)`** (lines 15-60)
   - Runs one conversation and evaluates it
   - Process:
     1. Creates ConversationManager
     2. Runs conversation
     3. Evaluates with JudgeAgent
     4. Prints results
     5. Saves to logs/
     6. Creates visualization
   - Returns full result dictionary

2. **`run_parameter_sweeps()`** (lines 62-130)
   - Runs systematic parameter sweeps
   - Tests:
     - Context sizes: [1000, 2000, 4000]
     - Max turns: [5, 10, 15]
     - Personality combinations: various pairs and all three
     - Claim types: sample from each category
   - Generates visualizations for each sweep
   - Saves all results
   - Prints best parameters found

3. **`run_baseline_comparison()`** (lines 132-180)
   - Compares baseline configuration vs optimized configuration
   - Baseline: 2 agents, 5 turns, 1000 context
   - Optimized: 3 agents, 10 turns, 2000 context
   - Tests on sample claims
   - Prints comparison results

4. **`main()`** (lines 182-210)
   - Interactive menu system
   - Prompts user for mode selection
   - Routes to appropriate function
   - Modes:
     - `single`: Single conversation
     - `sweep`: Parameter sweeps
     - `baseline`: Baseline comparison
     - `all`: Run everything

**Usage**:
```bash
python main.py              # Interactive menu
python main.py single       # Direct mode selection
python main.py sweep
python main.py baseline
python main.py all
```

### `example.py`

**Purpose**: Simple example script demonstrating basic usage.

**Key Function**: `example_single_conversation()` (lines 5-40)
- Loads a debatable claim
- Creates ConversationManager with personalities (can include deceiver to test misinformation spread)
- Runs short conversation (6 turns)
- Evaluates with JudgeAgent
- Prints results

**Usage**:
```bash
python example.py
```

---

## Data Flow

### Complete System Flow

```
1. User runs main.py or example.py
   ↓
2. Load claim from dataset
   ↓
3. Create ConversationManager with personalities
   ↓
4. For each turn:
   a. Generate prompt for agent (with personality + context)
   b. Call OpenAI API
   c. Get agent response
   d. Track stance
   e. Add to conversation history
   ↓
5. Conversation complete → Get final responses
   ↓
6. JudgeAgent evaluates:
   a. Build evaluation prompt
   b. Call OpenAI API
   c. Parse evaluation
   d. Return correctness, convergence, hallucinations
   ↓
7. Save results to logs/
   ↓
8. Visualizer creates plots
   ↓
9. Save visualizations to visualizations/
```

### Parameter Sweep Flow

```
1. Create ParameterSweep instance
   ↓
2. For each parameter value:
   a. Create ConversationManager with that parameter
   b. Run conversation
   c. Evaluate with JudgeAgent
   d. Record result
   ↓
3. Analyze results → Find best parameters
   ↓
4. Generate visualizations
   ↓
5. Save all results to results/
```

---

## Key Classes and Functions

### Core Classes

1. **`ConversationManager`** (`conversation/conversation_manager.py`)
   - Manages multi-agent conversations
   - Tracks stance changes
   - Coordinates agent interactions

2. **`JudgeAgent`** (`judge/judge_agent.py`)
   - Evaluates conversation correctness
   - Detects hallucinations
   - Assesses convergence

3. **`ParameterSweep`** (`evaluation/parameter_sweep.py`)
   - Systematically tests parameters
   - Analyzes results
   - Finds optimal configurations

4. **`Visualizer`** (`visualization/visualizer.py`)
   - Creates visualizations
   - Generates plots and charts

5. **`Claim`** (`data/dataset.py`)
   - Represents a claim to be debated
   - Contains text, type, ground truth

### Key Functions

1. **`get_claims_by_type(claim_type)`** (`data/dataset.py`)
   - Returns claims of specific type

2. **`get_personality_config(personality)`** (`agents/agent_factory.py`)
   - Returns personality configuration

3. **`run_conversation(claim)`** (`ConversationManager`)
   - Main conversation orchestration

4. **`evaluate_conversation(...)`** (`JudgeAgent`)
   - Evaluates conversation correctness

5. **`sweep_context_size(...)`** (`ParameterSweep`)
   - Tests different context sizes

---

## Configuration Customization

### Changing Models

Edit `config.py`:
```python
DEFAULT_MODEL = "gpt-4"  # or "gpt-3.5-turbo", etc.
```

### Adding New Personalities

Edit `config.py`, add to `AGENT_PERSONALITIES`:
```python
"new_personality": {
    "role": "Role Name",
    "goal": "Agent's goal",
    "backstory": "Detailed personality description",
    "personality_traits": ["trait1", "trait2"]
}
```

### Adding New Claims

Edit `data/dataset.py`, add to `DATASET`:
```python
ClaimType.GROUND_TRUTH: [
    ...existing claims...,
    Claim("Your new claim here.", ClaimType.GROUND_TRUTH)
]
```

### Adjusting Parameters

Edit `config.py`:
- `DEFAULT_TEMPERATURE`: Model creativity (0.0-1.0)
- `DEFAULT_MAX_TURNS`: Conversation length
- `DEFAULT_CONTEXT_SIZE`: Token limit
- `JUDGE_TEMPERATURE`: Judge consistency (lower = more consistent)

---

## Error Handling

The system includes error handling for:
- Missing API keys (raises clear error)
- API connection failures (returns error message)
- Invalid personalities (validates and raises error)
- File I/O errors (creates directories automatically)
- Import errors (clear error messages)

---

## Dependencies

All dependencies are listed in `requirements.txt`:
- `openai`: OpenAI API client
- `python-dotenv`: Environment variable management
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `tqdm`: Progress bars
- `pydantic`: Data validation

---

## Output Files

### Logs (`logs/`)
- JSON files with full conversation results
- Filename format: `conversation_YYYYMMDD_HHMMSS.json`
- Contains: conversation history, stance tracking, evaluation

### Results (`results/`)
- JSON files from parameter sweeps
- Filename format: `parameter_sweep_results_YYYYMMDD_HHMMSS.json`
- Contains: all parameter test results, best parameters

### Visualizations (`visualizations/`)
- PNG image files
- Stance shift plots
- Accuracy vs parameter charts
- Convergence heatmaps
- Hallucination detection charts

---

## Extending the System

### Adding New Evaluation Metrics

1. Modify `JudgeAgent.evaluate_conversation()` to include new metrics
2. Update `_parse_evaluation()` to extract new metrics
3. Update visualization functions to plot new metrics

### Adding New Parameter Sweeps

1. Add new method to `ParameterSweep` class
2. Follow pattern of existing sweep methods
3. Add visualization in `main.py`

### Adding New Visualization Types

1. Add new method to `Visualizer` class
2. Use matplotlib/seaborn for plotting
3. Save to `visualizations/` directory
4. Call from `main.py` or `example.py`

---

## Summary

This project implements a sophisticated multi-agent conversation system with:
- **Modular architecture**: Each component in separate module
- **Configurable agents**: Four distinct personalities (including a deceiver agent to test misinformation spread)
- **Comprehensive evaluation**: Judge agent with multiple metrics
- **Systematic testing**: Parameter sweep system
- **Rich visualizations**: Multiple plot types
- **Extensible design**: Easy to add features

The system explores how misinformation spreads in multi-agent conversations and identifies factors that lead to hallucinations or convergence to correct answers.

