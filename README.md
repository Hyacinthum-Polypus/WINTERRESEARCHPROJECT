# Winter Research Project - Dev Container

This repository is configured with a VS Code Dev Container for a reproducible Python environment.

## What gets installed

- Python 3.12 base image (Debian bullseye)
- System build tools (build-essential)
- Python dependencies pinned in `requirements.txt`
- Optional: LiteLLM + Postgres services via `.devcontainer/docker-compose.yml`

## Rebuild the Dev Container

Use VS Code command palette:

- “Dev Containers: Rebuild and Reopen in Container”

Or from the Command Palette: “Remote-Containers: Rebuild Container”.

The Dockerfile installs packages from `requirements.txt` during build.
After the container starts, `postCreateCommand` runs `pip install -r requirements.txt` to ensure any local changes are applied.

## Updating dependencies

- Edit `requirements.txt` and rebuild the container.
- If you update packages inside the container and want to pin them, run:

```bash
python3 -m pip freeze > requirements.txt
```

## Running

```bash
python3 kggen.py
```

If you’re using Ollama on your host, make sure it’s running and that `OLLAMA_BASE_URL` points to `http://host.docker.internal:11434` (default provided in `.env`).

## Troubleshooting

- If the Python script can’t reach Ollama, verify the host address and that the model is pulled locally.
- If LiteLLM server is needed, ensure the `litellm` service is running on port 4000 and adjust `LITELLM_BASE_URL` as needed.
