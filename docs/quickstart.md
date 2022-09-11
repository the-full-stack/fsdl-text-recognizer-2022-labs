# Quickstart

This document contains instructions _only_ to get a fully working development environment for
running this repo. For pre-requisites (e.g. `pyenv` install instructions) plus details on what's
being installed and why, please see [docs/getting_started.md](docs/getting_started.md).

We assume the following are installed and configured:
  - [pyenv](https://github.com/pyenv/pyenv)
  - [pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper)
  - [Poetry](https://python-poetry.org/docs/)
  - [zsh-autoswitch-virtualenv](https://github.com/MichaelAquilina/zsh-autoswitch-virtualenv)
  - [direnv](https://direnv.net/)


## Part 1: Generic Python setup

```sh
# Get the repo
git clone ${REPO_GIT_URL}

# Install Python
pyenv install $(cat .python-version)
pyenv shell $(cat .python-version)
python -m pip install --upgrade pip
python -m pip install virtualenvwrapper
pyenv virtualenvwrapper

# Setup the virtualenv
mkvirtualenv -p python$(cat .python-version) $(cat .venv)
python -V
python -m pip install --upgrade pip

# Install dependencies with Poetry
poetry self update
poetry install --no-root --sync

# Create templated .env for storing secrets
cp .env.template .env
direnv allow

# Create and audit secrets baseline
# N.B. Adjust the exclusions here depending on your needs (check .pre-commit-config.yaml)
detect-secrets --verbose scan \
    --exclude-files 'poetry\.lock' \
    --exclude-files '\.secrets\.baseline' \
    --exclude-files '\.env\.template' \
    --exclude-secrets 'password|ENTER_PASSWORD_HERE|INSERT_API_KEY_HERE' \
    --exclude-lines 'integrity=*sha' \
    > .secrets.baseline

detect-secrets audit .secrets.baseline
```


## Part 2: Project-specific setup

Please check [docs/project_specific_setup.md](docs/project_specific_setup.md) for further instructions.
