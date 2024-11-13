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


## Branching and Pushing Code to Repository

When working on an improvement or feature, please create a new branch with a prefix denoting your name and surname, and then the branch name. For example: `fy/improving_node_evolution`. This way, when you do `git branch -r`, you can see who has what branches open.

When the branch is ready for review, then make a Pull Request (PR) into GitHub and set someone to review it. If it's something trivial, then review it yourself and merge it. Please don't push directly into `main` (this will be blocked soon).

We will add some tests for core module functionality so that we won't be able to push to `main` without passing tests. This is an effort to prevent introducing breaking changes.

I also want to introduce automatic benchmark running so we always know the current score of the `main` branch.