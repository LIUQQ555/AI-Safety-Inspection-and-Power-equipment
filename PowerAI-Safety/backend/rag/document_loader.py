"""知识库文档加载与切分。

对应技术方案第八节：PDF/Word/标准文档 → 文档解析 → 文本切分。

支持格式：
    ``.pdf``  PyMuPDF（保留页码，便于引用溯源）
    ``.docx`` python-docx
    ``.txt`` / ``.md``  纯文本

切分策略：先按段落聚合到 ``chunk_size`` 字符，相邻块保留 ``chunk_overlap``
重叠，避免把一句完整的规定从中间截断导致检索语义丢失。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@dataclass
class Document:
    """一篇完整文档。"""

    doc_id: str
    title: str
    text: str
    source_path: str
    pages: List[str] = field(default_factory=list)   # 逐页文本，非分页文档为空


@dataclass
class Chunk:
    """切分后的检索单元。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: str = ""
    page: Optional[int] = None
    index: int = 0

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "source_path": self.source_path,
            "page": self.page,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            title=str(data.get("title", "")),
            text=str(data.get("text", "")),
            source_path=str(data.get("source_path", "")),
            page=data.get("page"),
            index=int(data.get("index", 0)),
        )


# ---------------------------------------------------------------------------
# 各格式解析
# ---------------------------------------------------------------------------
def _load_pdf(path: Path) -> tuple[str, List[str]]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "缺少 PyMuPDF，无法解析 PDF。安装：pip install pymupdf"
        ) from exc

    pages: List[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages.append(page.get_text())
    return "\n\n".join(pages), pages


def _load_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:
        raise ImportError(
            "缺少 python-docx，无法解析 Word。安装：pip install python-docx"
        ) from exc

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    # 表格内容往往是规程里的限值表，必须一并提取
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _load_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # 全部失败时用替换字符兜底，保证流程不中断
    return path.read_text(encoding="utf-8", errors="replace")


def load_document(path: Path) -> Optional[Document]:
    """加载单个文档，失败返回 None 并记录日志。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    try:
        if suffix == ".pdf":
            text, pages = _load_pdf(path)
        elif suffix == ".docx":
            text, pages = _load_docx(path), []
        else:
            text, pages = _load_text(path), []
    except Exception as exc:
        logger.error("解析文档失败 %s：%s", path, exc)
        return None

    text = text.strip()
    if not text:
        logger.warning("文档内容为空，已跳过：%s", path)
        return None

    return Document(
        doc_id=path.stem,
        title=path.stem,
        text=text,
        source_path=str(path),
        pages=pages,
    )


def load_directory(directory: Path, recursive: bool = True) -> List[Document]:
    """加载目录下所有支持的文档。"""
    directory = Path(directory)
    if not directory.exists():
        logger.warning("知识库目录不存在：%s", directory)
        return []

    pattern = "**/*" if recursive else "*"
    documents: List[Document] = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        document = load_document(path)
        if document is not None:
            documents.append(document)
            logger.info("已加载知识文档：%s（%d 字符）", path.name, len(document.text))
    return documents


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------
def _split_paragraphs(text: str) -> List[str]:
    """按空行拆段；无空行时按单换行拆。"""
    blocks = [block.strip() for block in text.split("\n\n")]
    result: List[str] = []
    for block in blocks:
        if not block:
            continue
        if len(block) <= 800:
            result.append(block)
        else:
            # 超长段落再按行拆，避免整段无法切分
            result.extend(line.strip() for line in block.split("\n") if line.strip())
    return result


def chunk_text(
    text: str,
    chunk_size: int = 480,
    chunk_overlap: int = 80,
    doc_id: str = "",
    title: str = "",
    source_path: str = "",
    page: Optional[int] = None,
    start_index: int = 0,
) -> List[Chunk]:
    """把长文本切分为带重叠的块。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    paragraphs = _split_paragraphs(text)
    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if content:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#{start_index + len(chunks)}",
                doc_id=doc_id,
                title=title,
                text=content,
                source_path=source_path,
                page=page,
                index=start_index + len(chunks),
            ))
        # 保留末尾若干字符作为下一块的开头，维持上下文连续
        if chunk_overlap > 0 and content:
            tail = content[-chunk_overlap:]
            buffer = [tail]
            buffer_len = len(tail)
        else:
            buffer = []
            buffer_len = 0

    for paragraph in paragraphs:
        # 单个段落本身超过 chunk_size：先冲掉缓存，再硬切
        if len(paragraph) > chunk_size:
            flush()
            buffer, buffer_len = [], 0
            for offset in range(0, len(paragraph), chunk_size - chunk_overlap):
                piece = paragraph[offset:offset + chunk_size].strip()
                if piece:
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}#{start_index + len(chunks)}",
                        doc_id=doc_id,
                        title=title,
                        text=piece,
                        source_path=source_path,
                        page=page,
                        index=start_index + len(chunks),
                    ))
            continue

        if buffer_len + len(paragraph) + 1 > chunk_size:
            flush()
        buffer.append(paragraph)
        buffer_len += len(paragraph) + 1

    # 收尾：flush 后 buffer 可能只剩重叠尾巴，若剩余内容过短则丢弃
    if buffer and len("\n".join(buffer).strip()) > chunk_overlap:
        flush()

    return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 480,
    chunk_overlap: int = 80,
) -> List[Chunk]:
    """对多篇文档切分。分页文档按页切分，保证页码可溯源。"""
    chunks: List[Chunk] = []
    for document in documents:
        if document.pages:
            for page_number, page_text in enumerate(document.pages, start=1):
                if not page_text.strip():
                    continue
                chunks.extend(chunk_text(
                    page_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    doc_id=document.doc_id,
                    title=document.title,
                    source_path=document.source_path,
                    page=page_number,
                    start_index=len(chunks),
                ))
        else:
            chunks.extend(chunk_text(
                document.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=document.doc_id,
                title=document.title,
                source_path=document.source_path,
                start_index=len(chunks),
            ))
    return chunks


__all__ = [
    "SUPPORTED_SUFFIXES",
    "Document",
    "Chunk",
    "load_document",
    "load_directory",
    "chunk_text",
    "chunk_documents",
]
