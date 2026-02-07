"""文件与 URL 的文档入库辅助方法。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from docx2txt import process as docx_to_text
from pypdf import PdfReader

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from .config import settings


def extract_text_from_pdf(path: Path) -> list[dict]:
    """提取 PDF 每一页的文本。

    Args:
        path: PDF 文件路径。

    Returns:
        包含文本与页码的列表。
    """
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"text": text, "page": i + 1})
    return pages


def extract_text_from_docx(path: Path) -> str:
    """提取 DOCX 文件文本。

    Args:
        path: DOCX 文件路径。

    Returns:
        提取出的文本。
    """
    return docx_to_text(str(path))


def extract_text_from_txt(path: Path) -> str:
    """提取 TXT 文件文本。

    Args:
        path: TXT 文件路径。

    Returns:
        提取出的文本。
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_url(url: str, timeout: int = 10) -> str:
    """抓取 URL 并提取可见文本。

    Args:
        url: 目标 URL。
        timeout: 请求超时（秒）。

    Returns:
        清洗后的页面文本。
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines())
    return "\n".join([line for line in text.splitlines() if line])


def build_documents_from_file(path: Path, source_ref: str) -> list[Document]:
    """从本地文件构建 LlamaIndex 文档。

    Args:
        path: 源文件路径。
        source_ref: 写入元数据的来源引用。

    Returns:
        文档列表。
    """
    ext = path.suffix.lower()
    docs: list[Document] = []
    if ext == ".pdf":
        for page in extract_text_from_pdf(path):
            docs.append(
                Document(
                    text=page["text"],
                    metadata={"source": source_ref, "page": page["page"]},
                )
            )
    elif ext == ".docx":
        text = extract_text_from_docx(path)
        docs.append(Document(text=text, metadata={"source": source_ref}))
    elif ext == ".txt":
        text = extract_text_from_txt(path)
        docs.append(Document(text=text, metadata={"source": source_ref}))
    else:
        raise ValueError("Unsupported file type")
    return docs


def build_documents_from_url(url: str) -> list[Document]:
    """从 URL 构建单个文档。

    Args:
        url: 来源 URL。

    Returns:
        仅包含一个文档的列表。
    """
    text = extract_text_from_url(url)
    return [Document(text=text, metadata={"source": url})]


def split_into_chunks(docs: Iterable[Document]) -> list[Document]:
    """将文档切分为可检索的分块节点。

    Args:
        docs: 文档迭代器。

    Returns:
        分块后的节点列表。
    """
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.get_nodes_from_documents(list(docs))
