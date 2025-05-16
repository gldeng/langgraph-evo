.PHONY: clean format lint test install dev docs

clean:
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black src tests examples
	isort src tests examples

lint:
	ruff src tests examples
	mypy src tests examples

test:
	pytest --cov=src tests/

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

docs:
	mkdir -p docs/api
	# Add documentation generation command when docs system is set up 