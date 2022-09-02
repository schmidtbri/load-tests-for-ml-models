.DEFAULT_GOAL := help

.PHONY: help clean-pyc build clean-build venv dependencies test-dependencies clean-venv test test-reports clean-test check-codestyle check-docstyle

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

venv: ## create virtual environment
	python3.9 -m venv venv

dependencies: ## install dependencies from requirements.txt
	python -m pip install --upgrade pip
	python -m pip install --upgrade setuptools
	python -m pip install --upgrade wheel
	pip install -r requirements.txt

test-dependencies: ## install dependencies from test_requirements.txt
	pip install -r test_requirements.txt

update-dependencies:  ## Update dependency versions
	pip-compile requirements.in > requirements.txt
	pip-compile test_requirements.in > test_requirements.txt
	pip-compile service_requirements.in > service_requirements.txt

clean-venv: ## remove all packages from virtual environment
	pip freeze | grep -v "^-e" | xargs pip uninstall -y

clean-test:	## Remove test artifacts
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf reports
	rm -rf .pytype

check-dependencies:  ## checks for security vulnerabilities in dependencies
	safety check -r requirements.txt

convert-post:  ## Convert the notebook into Markdown file
	jupyter nbconvert --to markdown blog_post/blog_post.ipynb --output-dir='./blog_post' --TagRemovePreprocessor.remove_input_tags='{"hide_code"}'

build-image:  ## Build docker image
	export BUILD_DATE=`date -u +'%Y-%m-%dT%H:%M:%SZ'` && \
	docker build --build-arg BUILD_DATE \
		-t insurance_charges_model_service:latest .