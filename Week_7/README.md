# AI Student Week 7

This repository contains implementations of various AI applications including basic LLM integration and RAG (Retrieval Augmented Generation) systems.

## Project Structure

- `basic_llm.py`: Implementation of basic large language model interaction
- `rag.py`: Retrieval Augmented Generation system implementation

## Setup Instructions

1. Clone this repository
2. Copy the environment template and fill in your API keys:
   ```bash
   cp .env.template .env
   ```
   Then edit the `.env` file to add your API keys.
3. Run the scripts with uv. E.g.
   ```bash
   uv run basic_llm.py
   ```

## Usage

### Basic LLM

Run the basic LLM implementation:

```bash
uv run basic_llm.py
```

### RAG System

Run the Retrieval Augmented Generation system:

```bash
uv run rag.py
```

## Requirements

- Python 3.8+ (see `.python-version`)
- Dependencies are managed with uv (see `uv.lock` and `pyproject.toml`)

## Configuration

The application uses environment variables for configuration (API keys, etc.). See `.env.template` for required variables. 