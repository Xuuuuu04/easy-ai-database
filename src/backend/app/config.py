from pydantic import BaseModel
import os
from pathlib import Path
from urllib.parse import urlparse


def _get_env_path() -> Path:
    return Path("/Users/xushaoyang/Desktop/ima-simple/.env")


def _load_env():
    env_path = _get_env_path()
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env()


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
            os.environ[key] = "socks5h://" + value[len("socks5://") :]


class Settings(BaseModel):
    data_dir: str = ""
    db_path: str = ""
    index_dir: str = ""

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    embed_batch_size: int = 96
    embed_num_workers: int = 8

    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""
    rerank_top_k: int = 10

    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.95
    cache_max_size: int = 1000
    cache_ttl_hours: int = 24

    enable_hybrid_search: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    enable_multi_turn: bool = True
    max_chat_history: int = 3
    agent_max_rounds: int = 3

    max_context_chunks: int = 6
    chunk_size: int = 512
    chunk_overlap: int = 64

    require_citations: bool = True
    min_citations: int = 1

    mock_mode: bool = False
    use_env_proxy: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        self.data_dir = os.getenv("DATA_DIR", "./data")
        self.db_path = os.getenv("DB_PATH", "./data/app.db")
        self.index_dir = os.getenv("INDEX_DIR", "./data/index")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY", "local-key")
        self.llm_model = os.getenv("LLM_MODEL", "local-model")
        self.embed_base_url = os.getenv("EMBED_BASE_URL", "http://localhost:1234/v1")
        self.embed_api_key = os.getenv("EMBED_API_KEY", "local-key")
        self.embed_model = os.getenv("EMBED_MODEL", "local-embedding")
        self.embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "96"))
        self.embed_num_workers = int(os.getenv("EMBED_NUM_WORKERS", "8"))
        self.max_context_chunks = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
        self.require_citations = os.getenv("REQUIRE_CITATIONS", "1") == "1"
        self.min_citations = int(os.getenv("MIN_CITATIONS", "1"))
        self.mock_mode = os.getenv("MOCK_MODE", "0") == "1"
        self.use_env_proxy = os.getenv("USE_ENV_PROXY", "1") == "1"

        self.rerank_base_url = os.getenv("RERANK_BASE_URL", self.llm_base_url)
        self.rerank_api_key = os.getenv("RERANK_API_KEY", self.llm_api_key)
        self.rerank_model = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "10"))

        self.enable_semantic_cache = os.getenv("ENABLE_SEMANTIC_CACHE", "1") == "1"
        self.cache_similarity_threshold = float(
            os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95")
        )
        self.cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        self.cache_ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))

        self.enable_hybrid_search = os.getenv("ENABLE_HYBRID_SEARCH", "1") == "1"
        self.bm25_weight = float(os.getenv("BM25_WEIGHT", "0.3"))
        self.vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.7"))

        self.enable_multi_turn = os.getenv("ENABLE_MULTI_TURN", "1") == "1"
        self.max_chat_history = int(os.getenv("MAX_CHAT_HISTORY", "3"))
        self.agent_max_rounds = int(os.getenv("AGENT_MAX_ROUNDS", "3"))


settings = Settings()

_ensure_no_proxy_for_local(settings.llm_base_url)
_ensure_no_proxy_for_local(settings.embed_base_url)
_ensure_no_proxy_for_local(settings.rerank_base_url)
_normalize_socks_proxy()
