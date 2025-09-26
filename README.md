# Winter Research Project - Dev Container

This repository is configured with a VS Code Dev Container for a reproducible Python environment.

## What gets installed

- Python 3.12 base image (Debian bullseye)
- System build tools (build-essential)
- Python dependencies pinned in `requirements.txt`
- Optional: LiteLLM + Postgres services via `.devcontainer/docker-compose.yml`
 - Optional: ArangoDB service for graph/knowledge storage

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

### ArangoDB usage

An ArangoDB service is available in the dev container stack:

- URL inside containers: `http://arangodb:8529`
- URL from host: `http://localhost:8529`
- Root password: `password`

From Python (inside the dev container), `python-arango` is already in `requirements.txt`:

```python
from arango import ArangoClient

client = ArangoClient(hosts="http://arangodb:8529")
sys_db = client.db("_system", username="root", password="password")

# Create or get a database
if not sys_db.has_database("winterresearch"):
	sys_db.create_database("winterresearch")

db = client.db("winterresearch", username="root", password="password")
graph_name = "kg"
if not db.has_graph(graph_name):
	db.create_graph(graph_name)

print("ArangoDB connected and graph ensured.")
```

ArangoDB web UI (ArangoDB Web Interface) is exposed at http://localhost:8529 with user `root` / `password`.

## Troubleshooting

- If the Python script can’t reach Ollama, verify the host address and that the model is pulled locally.
- If LiteLLM server is needed, ensure the `litellm` service is running on port 4000 and adjust `LITELLM_BASE_URL` as needed.
 - If ArangoDB is unreachable, verify the service is healthy with `docker ps` and that port `8529` is forwarded. Inside the devcontainer, reach it at `http://arangodb:8529`.
