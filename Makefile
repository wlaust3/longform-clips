.PHONY: install lint fmt test check clean
install:
	python -m pip install -e ".[dev]"
lint:
	ruff check .
	ruff format --check .
fmt:
	ruff check --fix .
	ruff format .
test:
	pytest -q
check: lint test
clean:
	rm -rf out .pytest_cache .ruff_cache build dist *.egg-info
