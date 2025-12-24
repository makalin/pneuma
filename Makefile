# Makefile for PNEUMA

# Use bash for all recipes
SHELL := /bin/bash

.PHONY: all install clean run train test lint

all: install

# ==============================================================================
# INSTALLATION & DEPENDENCIES
# ==============================================================================

install:
	@echo "--- Installing Pneuma in editable mode ---"
	pip install -e .

# ==============================================================================
# DEVELOPMENT
# ==============================================================================

run:
	@echo "--- Running PNEUMA ---"
	pneuma

# Example: make run-persona persona=heckler
run-persona:
	@echo "--- Running PNEUMA with persona: $(persona) ---"
	pneuma --persona=$(persona)

train:
	@echo "--- Starting training ritual ---"
	python train_voice.py

# Example: make train-epochs epochs=1000
train-epochs:
	@echo "--- Starting training ritual for $(epochs) epochs ---"
	python train_voice.py --epochs=$(epochs)

# ==============================================================================
# TESTING & QUALITY
# ==============================================================================

test:
	@echo "--- Running tests ---"
	pytest

lint:
	@echo "--- Linting with ruff ---"
	ruff check .

# =================================="
# CLEANUP
# =================================="

clean:
	@echo "--- Cleaning up bytecode and cache ---"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

