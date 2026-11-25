PY ?= python
.PHONY: setup demo download pipeline tiles train infer change aggregate serve serve-demo test mlflow docker

setup:        ## install dependencies (inside your conda env)
	$(PY) -m pip install -r requirements.txt
demo:         ## full pipeline on synthetic data, offline
	$(PY) -m greenwatch demo
download:     ## Sentinel-2 composites + WorldCover labels (internet, ~GBs)
	$(PY) -m greenwatch download
pipeline:     ## tiles -> train -> infer -> change -> aggregate on real data
	$(PY) -m greenwatch pipeline
tiles train infer change aggregate:
	$(PY) -m greenwatch $@
serve:
	$(PY) -m greenwatch serve
serve-demo:
	GW_OUTPUTS=outputs/demo $(PY) -m greenwatch serve
test:
	$(PY) -m pytest -q
mlflow:
	mlflow ui --port 5000
docker:
	docker compose up --build
