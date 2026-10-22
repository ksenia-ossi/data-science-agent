.PHONY: install test lint demo docker-build docker-demo

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check ds_agent tests examples
	.venv/bin/ruff format --check ds_agent tests examples

data/telco_churn.csv:
	.venv/bin/python examples/download_churn.py

demo: data/telco_churn.csv
	.venv/bin/ds-agent data/telco_churn.csv --target Churn \
		--task "Predict which customers will churn, so the retention team can contact the riskiest ones first."

docker-build:
	docker build -t ds-agent .

docker-demo: docker-build data/telco_churn.csv
	docker run --rm -it -e ANTHROPIC_API_KEY -v "$(PWD)":/work ds-agent \
		data/telco_churn.csv --target Churn \
		--task "Predict which customers will churn, so the retention team can contact the riskiest ones first."
