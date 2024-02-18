# Makefile for Python project

VENV_DIR := venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest
SRC_DIR := src
TESTS_DIR := tests

# Help target
help:
	@echo "Available commands:"
	@echo "  setup   : Sets up virtual environment and installs dependencies"
	@echo "  test    : Runs all tests using pytest"
	@echo "  run     : Runs the project"
	@echo "  clean   : Removes the virtual environment"

# Set up virtual environment and install dependencies
setup: $(VENV_DIR)/bin/activate requirements.txt
	. $(VENV_DIR)/bin/activate; \
	$(PIP) install -r requirements.txt

# Create virtual environment if it doesn't exist
$(VENV_DIR)/bin/activate:
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)

# Run all tests using pytest
test: setup
	. $(VENV_DIR)/bin/activate; \
	$(PYTEST) $(TESTS_DIR)

# Run the project
run:
	$(PYTHON) $(SRC_DIR)/main.py

# Clean up
clean:
	rm -rf $(VENV_DIR)
