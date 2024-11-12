# the-greatest-team

ECS 170 Final Project

Dataset: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download)

## Environment Setup Instructions

To set up the development environment for this project, please follow the steps below:

### 1. Install OpenMP (macOS Users Only)

If you're using macOS, you may need to install OpenMP to ensure compatibility with certain dependencies:

```bash
brew install libomp

```

2. Install uv
If you don't have uv installed, you can install it using one of the following methods:

Using pip:

```bash 
pip install uv
```
Using the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and Activate a Virtual Environment
To create and activate a virtual environment for the project:

```bash
uv venv
source .venv/bin/activate
```
4. Install Project Dependencies

```bash
uv pip sync requirements.txt
```
This command will install all the packages listed in the requirements.txt file.

1. Add Additional Dependencies
To add new packages to the project, use the following command:

```bash
uv add <package-name>
```
Replace <package-name> with the name of the package you wish to add. This command will add the package to your pyproject.toml file and install it into the virtual environment