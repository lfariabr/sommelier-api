VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install train api ui test lint freeze clean

install:  ## create venv (needs Python 3.11.x — see .python-version) + install deps
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

train:  ## reproduce both models -> ml/artifacts/
	$(PY) -m ml.train

api:  ## run the FastAPI service at http://localhost:8000/docs
	$(PY) -m uvicorn api.main:app --reload --port 8000

ui:  ## run the Streamlit app at http://localhost:8501
	$(VENV)/bin/streamlit run ui/app.py

test:  ## run the test suite
	$(PY) -m pytest -q

lint:  ## ruff lint
	$(VENV)/bin/ruff check .

freeze:  ## snapshot the resolved environment
	$(PIP) freeze > requirements.lock.txt

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
