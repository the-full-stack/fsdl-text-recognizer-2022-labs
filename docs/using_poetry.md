# Updating Python requirements using Poetry

This project uses Poetry to manage dependencies:
[https://python-poetry.org/docs/](https://python-poetry.org/docs/)

The Poetry virtualenv should activate automatically if you are using
[zsh-autoswitch-virtualenv](https://github.com/MichaelAquilina/zsh-autoswitch-virtualenv). You can
also activate it manually by running `workon fsdl` (preferred) or `poetry shell` (Poetry
created/managed virtualenv).


## Poetry tl;dr

```bash
# Sync your environment with poetry.lock
poetry install --no-root --sync

# Add and install a new package into your environment (or update an existing one)
# 1. Add a package to pyproject.toml
# 2. Run this
poetry add package@version

# N.B. We lock & export to requirements when you run `pre-commit run --all-files` so you
# shouldn't need to run the commands below yourself.

# Compile all poetry.lock file
poetry update

# Export to requirements.txt (this is for compatibility with team members who may not
# have Poetry; Poetry itself uses poetry.lock when you run `poetry install`)
poetry export -f requirements.txt --output requirements.txt
```


## Poetry FAQ

**How do I sync my environment with the latest `poetry.lock`?**
To install dependencies, simply `cd backend-security` and run `poetry install --no-root --sync`.
This will resolve & install all deps in `pyproject.toml` via `poetry.lock`. Options:
  - `--no-root` skips installing the repo itself as a Python package
  - `--sync` removes old dependencies no longer present in the lock file

You may wish to set an alias e.g. `alias poetry-sync='poetry install --no-root --sync'`

**How do I add/change packages?**
In order to install a new package or remove an old one, please edit `pyproject.toml`
or use either `poetry add package` or `pipx run poetry add package@version` as desired.

**How do I update `poetry.lock` to match new changes in `pyproject.toml`?**
You can run `poetry update`, although this will also get run when you run pre-commit. This fetches
the latest matching versions (according to `pyproject.toml`) and updates `poetry.lock`, and is
equivalent to deleting `poetry.lock` and running `poetry install --no-root` again from scratch.

**What do `~`, `^` and `*` mean?**
Check out the Poetry docs page for [specifying version constraints](https://python-poetry.org/docs/dependency-specification/#version-constraints).
Environment markers are great, e.g. this means "for Python 2.7.* OR Windows, install pathlib2 >=2.2, <3.0"
`pathlib2 = { version = "^2.2", markers = "python_version ~= '2.7' or sys_platform == 'win32'" }`
