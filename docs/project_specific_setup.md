# Project-specific setup

This document contains setup instructions specifically for this project only. This design enables
us to keep other docs easily aligned with future upstream changes to
[coefficient-cookiecutter](https://github.com/CoefficientSystems/coefficient-cookiecutter/).


## Install system-level dependencies with [homebrew](https://brew.sh/)

```sh
# brew install wkhtmltopdf
```

## gcloud setup

Install [gcloud as per these instructions](https://cloud.google.com/sdk/docs/install).

Run `gcloud init` to create a new named configuration. Unless it exists already, create a new
configuration as required, e.g. `fsdl`. This configuration can be activated at any
point, for example:

```sh
gcloud config set project fsdl

# Top tip: install tldr to get quick help with gcloud commands!
# https://dbrgn.github.io/tealdeer/
brew install tealdeer
tldr gcloud
```


## Jupyter kernel

```sh
python -m ipykernel install --user --name fsdl --display-name "Python (fsdl)"
```
