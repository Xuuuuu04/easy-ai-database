from pydantic import BaseModel
import os
from urllib.parse import urlparse


def _ensure_no_proxy_for_local(url: str) -> None:
    try:
        host = urlparse(url).hostname
    except Exception:
        host = None
    if host in {"localhost", "127.0.0.1"}:
        current = os.getenv("NO_PROXY", "")
        entries = [e for e in current.split(",") if e]
        for target in ("localhost", "127.0.0.1"):
            if target not in entries:
                entries.append(target)
        os.environ["NO_PROXY"] = ",".join(entries)

def _normalize_socks_proxy() -> None:
    # Prefer explicit HTTP(S) proxies over ALL_PROXY socks settings
    http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    all_proxy = os.environ.get("ALL_PROXY")
    if all_proxy and all_proxy.startswith("socks5://") and (http_proxy or https_proxy):
        os.environ.pop("ALL_PROXY", None)

    # httpx socks needs socks5h for remote DNS in many environments
    for key in ("ALL_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        value = os.environ.get(key)
        if not value:
            continue
        if value.startswith("socks5://"):
            os.environ[key] = "socks5h://" + value[len("socks5://"):]


class Settings(BaseModel):
    data_dir: str = os.getenv("DATA_DIR", "./data")
    db_path: str = os.getenv("DB_PATH", "./data/app.db")
    index_dir: str = os.getenv("INDEX_DIR", "./data/index")

    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "local-key")
    llm_model: str = os.getenv("LLM_MODEL", "local-model")

    embed_base_url: str = os.getenv("EMBED_BASE_URL", "http://localhost:1234/v1")
    embed_api_key: str = os.getenv("EMBED_API_KEY", "local-key")
    embed_model: str = os.getenv("EMBED_MODEL", "local-embedding")

    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    require_citations: bool = os.getenv("REQUIRE_CITATIONS", "1") == "1"
    min_citations: int = int(os.getenv("MIN_CITATIONS", "1"))

    mock_mode: bool = os.getenv("MOCK_MODE", "0") == "1"
    use_env_proxy: bool = os.getenv("USE_ENV_PROXY", "1") == "1"


settings = Settings()

_ensure_no_proxy_for_local(settings.llm_base_url)
_ensure_no_proxy_for_local(settings.embed_base_url)
_normalize_socks_proxy()
