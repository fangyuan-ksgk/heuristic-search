# Variables
PROJECT_NAME = nirvana 
VENV = .venv

# Default target
all: install

# Create a virtual environment
$(VENV)/bin/activate:
	python -m venv $(VENV)

# Install dependencies
install: $(VENV)/bin/activate
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -e .

# Run tests
test:
	$(VENV)/bin/pytest tests/

# Run linter (e.g., flake8 for code style)
lint:
	$(VENV)/bin/flake8 $(PROJECT_NAME)

# Format code (e.g., black for code formatting)
format:
	$(VENV)/bin/black $(PROJECT_NAME)

# Clean up generated files
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Remove all generated files and directories
dist-clean: clean
	rm -rf build dist

# Run Jupyter notebook
notebook:
	$(VENV)/bin/jupyter notebook

# Help
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  install       Install project dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linter"
	@echo "  format        Format code"
	@echo "  clean         Clean generated files"
	@echo "  dist-clean    Clean all build and generated files"
	@echo "  notebook      Run Jupyter notebook"
	@echo "  help          Show this help message"
