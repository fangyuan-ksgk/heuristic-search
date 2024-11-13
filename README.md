# Evolution-of-Plan

Let LLM evolves its plan until it works ~
![Evolution-of-Plan](https://github.com/user-attachments/assets/af98faeb-66d6-4278-af86-67d668d1954e)

## Project Setup

The easiest way to set up the development environment is to use the provided Makefile:

```
make install
```

This will automatically create a virtual environment and install the project dependencies.

Alternatively, you can set up the environment manually:

## Virtual Environment Setup

To set up the development environment, first create a virtual environment:

```
python -m venv .venv
```

Then activate the virtual environment:

- On macOS/Linux: `source .venv/bin/activate`

## Project Installation

With the virtual environment activated, install the project dependencies:

```
pip install -e .
```

## Environment Variables

This project requires an API key stored in an `.env` file. Create the `.env` file in the root of the project directory and add the following:

```
OPENAI_API_KEY="xxxxxxxxxxxxxxxx"
ANTHROPIC_API_KEY="xxxxxxxxxxxxxxxx"
GROQ_API_KEY="xxxxxxxxxxxxx"
```

Make sure to never commit the `.env` file to version control, as it contains sensitive information.