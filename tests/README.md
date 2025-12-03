# Test Suite

This directory contains unit tests for the multi-agent conversation system.

## Running Tests

### Install test dependencies

```bash
pip install -r requirements.txt
```

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_dataset.py
pytest tests/test_judge_agent.py
pytest tests/test_advanced_metrics.py
```

### Run with coverage

```bash
pytest --cov=. --cov-report=html
```

### Run with verbose output

```bash
pytest -v
```

### Run specific test

```bash
pytest tests/test_dataset.py::TestClaim::test_claim_creation
```

## Test Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_dataset.py` - Tests for dataset and Claim classes
- `test_agent_factory.py` - Tests for agent personality configurations
- `test_judge_agent.py` - Tests for judge agent and accuracy verification
- `test_advanced_metrics.py` - Tests for conversation quality metrics
- `test_parameter_sweep.py` - Tests for parameter sweep functionality

## Test Coverage

The test suite covers:

- ✅ Dataset and Claim classes
- ✅ Agent factory functions
- ✅ Judge agent cosine similarity calculations
- ✅ Judge accuracy verification (with mocks)
- ✅ Advanced metrics calculations
- ✅ Parameter sweep basic functionality

## Note on API-Dependent Tests

Some components (like `ConversationManager`) require OpenAI API calls. These are not fully tested in the unit test suite to avoid:
- API costs
- Network dependencies
- Rate limiting issues

For integration testing with real API calls, see the example scripts in the root directory.

## Adding New Tests

When adding new functionality:

1. Create a test file: `tests/test_<module_name>.py`
2. Import the module and create test classes
3. Use fixtures from `conftest.py` when possible
4. Mock external dependencies (API calls, file I/O)
5. Test both success and error cases
6. Run tests before committing: `pytest`

