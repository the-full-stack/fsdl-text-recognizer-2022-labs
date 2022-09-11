# Lambda Labs Setup + VS Code Integration

## Create an SSH key on Lambda and clone the repo

Do the following in a terminal (e.g. via SSH, or using a Jupyter Terminal) on the remote machine.

```sh
ssh-keygen -t ed25519 -C "2884159+john-sandall@users.noreply.github.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add the new key here: [https://github.com/settings/keys]

```sh
git clone git@github.com:john-sandall/fsdl-text-recognizer-2022-labs.git
```

## Jupyter Terminal

We'll first install Miniconda, then use it to install Python 3.10 & create a new environment.

```sh
# Install OS-level dependencies
sudo apt-get update
sudo apt-get -y install direnv

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh

# Install Python + create environment
cd fsdl-text-recognizer-2022-labs
conda create --name $(cat .venv) python=$(cat .python-version) --no-default-packages
conda activate $(cat .venv)

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

Add this to `~/.bash_profile`:

```bash
export PATH="/home/jupyter/.local/bin:$PATH"
conda activate jp-data
cd ~/data
eval "$(direnv hook bash)"
```

```sh
source ~/.bash_profile
poetry --version
poetry self update
poetry install --no-root --sync

# Create templated .env for storing secrets
cp .env.template .env
direnv allow

# Create Jupyter kernel
python -m ipykernel install --user --name jp-data --display-name "Python (jp-data)"
```
