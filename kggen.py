import os
from kg_gen import KGGen
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI-compatible base URL for LiteLLM so KGGen routes via LiteLLM.
# Default to the docker-compose service name when running inside the devcontainer.
# Prefer explicit base URL from env; default to host's LiteLLM port (works from devcontainer)
_base_url = os.getenv("LITELLM_BASE_URL", "http://host.docker.internal:4000")
os.environ.setdefault("OPENAI_API_BASE", _base_url)
os.environ.setdefault("OPENAI_BASE_URL", _base_url)

# Ensure OPENAI_API_KEY is set for OpenAI-compatible clients
_api_key = os.getenv("LITELLM_API_KEY")
if _api_key:
    os.environ.setdefault("OPENAI_API_KEY", _api_key)

# Allow LiteLLM to locate Ollama if needed (when using ollama/ollama_chat providers)
_ollama_base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
os.environ.setdefault("OLLAMA_BASE_URL", _ollama_base)


kg = KGGen(
    # Use Ollama provider directly via LiteLLM Python client (chat endpoint)
    model="ollama_chat/gpt-oss:20b",
    temperature=0.0,
    api_key=os.getenv("LITELLM_MASTER_KEY") or os.getenv("LITELLM_API_KEY"),
    api_base=_ollama_base,
)

text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."

graph_1 = kg.generate(
    input_data=text_input,
    context="Family relationships"
)

print("Entities:", sorted(list(graph_1.entities)))
print("Edges:", sorted(list(graph_1.edges)))
print("Relations (sample up to 10):", list(graph_1.relations)[:10])