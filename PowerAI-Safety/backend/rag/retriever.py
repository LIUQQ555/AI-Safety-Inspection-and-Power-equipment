"""知识检索。

对应技术方案第八节：向量数据库 → 相关标准检索 → LLM 生成最终解释。

实现上使用 **numpy 余弦检索**：知识库规模为几百到几千个文本块，
全量内积在这个量级上耗时不到 1 毫秒，无需引入 FAISS。
若后续知识库增长到十万块以上，可替换为 FAISS —— 替换点仅在
``KnowledgeBase._load_index`` 与 ``search`` 两处，接口不变。

索引落盘为三个文件：
    ``chunks.jsonl``  文本块
    ``vectors.npy``   向量矩阵（行已 L2 归一化）
    ``embedder.pkl``  向量化器（含词表，必须与 vectors 配套）
    ``meta.json``     构建签名与统计信息
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.core.schemas import StandardReference
from backend.rag.document_loader import Chunk, chunk_documents, load_directory
from backend.rag.embedding import BaseEmbedder, create_embedder

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """电力标准/规程知识库。"""

    def __init__(self, config) -> None:
        self.config = config
        cfg = config.rag

        self.knowledge_dir: Path = config.resolve("rag", "knowledge_dir")
        self.index_dir: Path = config.resolve("rag", "index_dir")
        self.chunk_size = int(cfg.get("chunk_size", 480))
        self.chunk_overlap = int(cfg.get("chunk_overlap", 80))
        self.default_top_k = int(cfg.get("top_k", 4))
        self.min_score = float(cfg.get("min_score", 0.05))

        self.embedder: Optional[BaseEmbedder] = None
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    def _signature(self) -> str:
        """知识库内容签名：文件名 + 大小 + 修改时间。内容变化即触发重建。"""
        if not self.knowledge_dir.exists():
            return "empty"
        entries = []
        for path in sorted(self.knowledge_dir.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                stat = path.stat()
                entries.append(f"{path.relative_to(self.knowledge_dir)}:{stat.st_size}:{int(stat.st_mtime)}")
        digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()[:16]
        return f"{digest}:{len(entries)}"

    @property
    def index_exists(self) -> bool:
        return all((self.index_dir / name).exists() for name in
                   ("chunks.jsonl", "vectors.npy", "embedder.pkl", "meta.json"))

    def build(self, force: bool = False) -> int:
        """构建（或按需重建）索引，返回块数量。"""
        signature = self._signature()

        if self.index_exists and not force:
            meta = self._read_meta()
            if meta.get("signature") == signature:
                self._load_index()
                logger.info("知识库索引已是最新，共 %d 块", len(self.chunks))
                return len(self.chunks)
            logger.info("知识库内容已变化，重建索引")

        documents = load_directory(self.knowledge_dir)
        if not documents:
            logger.warning(
                "知识库目录 %s 中没有可用文档（支持 .pdf/.docx/.txt/.md）。"
                "RAG 检索将返回空结果，报告中的「依据标准」部分会留空。",
                self.knowledge_dir,
            )
            self.chunks, self.vectors = [], None
            self._write_empty_index(signature)
            return 0

        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            self._write_empty_index(signature)
            return 0

        self.embedder = create_embedder(self.config)
        vectors = self.embedder.fit_transform([chunk.text for chunk in chunks])

        self.chunks = chunks
        self.vectors = vectors
        self._save_index(signature)

        logger.info(
            "知识库构建完成：%d 篇文档 → %d 个文本块，向量维度 %d",
            len(documents), len(chunks), vectors.shape[1],
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # 索引读写
    # ------------------------------------------------------------------
    def _read_meta(self) -> Dict:
        path = self.index_dir / "meta.json"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("索引元信息读取失败：%s", exc)
            return {}

    def _save_index(self, signature: str) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

        with (self.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        np.save(self.index_dir / "vectors.npy", self.vectors)
        if self.embedder is not None:
            self.embedder.save(self.index_dir)

        meta = {
            "signature": signature,
            "n_chunks": len(self.chunks),
            "n_docs": len({chunk.doc_id for chunk in self.chunks}),
            "dim": int(self.vectors.shape[1]) if self.vectors is not None else 0,
            "embedder": self.embedder.name if self.embedder else "none",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with (self.index_dir / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    def _write_empty_index(self, signature: str) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "chunks.jsonl").write_text("", encoding="utf-8")
        meta = {
            "signature": signature, "n_chunks": 0, "n_docs": 0,
            "dim": 0, "embedder": "none",
            "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
        }
        with (self.index_dir / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    def _load_index(self) -> None:
        self.chunks = []
        chunks_path = self.index_dir / "chunks.jsonl"
        if chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.chunks.append(Chunk.from_dict(json.loads(line)))

        vectors_path = self.index_dir / "vectors.npy"
        self.vectors = np.load(vectors_path) if vectors_path.exists() else None

        try:
            self.embedder = create_embedder_from_index(self.config, self.index_dir)
        except Exception as exc:
            logger.error("向量化器加载失败，需重建索引：%s", exc)
            self.embedder = None
            self.vectors = None
            self.chunks = []

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.vectors is not None and self.embedder is not None

    def ensure_ready(self) -> bool:
        """确保索引可用；未构建时自动构建。"""
        if self.is_ready:
            return True
        try:
            self.build()
        except Exception as exc:
            logger.error("知识库构建失败：%s", exc)
            return False
        return self.is_ready

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[StandardReference]:
        """检索相关标准条款。索引不可用时返回空列表（不抛异常）。"""
        if not query or not query.strip():
            return []
        if not self.ensure_ready():
            return []

        top_k = top_k or self.default_top_k
        min_score = self.min_score if min_score is None else min_score

        query_vector = self.embedder.transform([query])[0]
        scores = self.vectors @ query_vector          # 向量已归一化，内积即余弦

        top_k = max(1, min(top_k, len(scores)))
        # argpartition 取前 k，再对这 k 个排序，避免全量排序
        candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

        references: List[StandardReference] = []
        for idx in candidate_idx:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = self.chunks[int(idx)]
            references.append(StandardReference(
                doc_id=chunk.doc_id,
                title=chunk.title,
                snippet=_truncate(chunk.text, 400),
                score=round(score, 4),
                source_path=chunk.source_path,
                page=chunk.page,
            ))
        return references

    def stats(self) -> Dict:
        meta = self._read_meta()
        return {
            "ready": self.is_ready,
            "n_chunks": len(self.chunks) if self.chunks else int(meta.get("n_chunks", 0)),
            "n_docs": int(meta.get("n_docs", 0)),
            "dim": int(meta.get("dim", 0)),
            "embedder": meta.get("embedder", "none"),
            "knowledge_dir": str(self.knowledge_dir),
            "index_dir": str(self.index_dir),
        }


def create_embedder_from_index(config, index_dir: Path) -> BaseEmbedder:
    """加载与索引配套的向量化器。

    必须用索引构建时的同一个向量化器（词表一致），否则查询向量与
    库中向量的坐标含义不同，相似度完全失效。
    """
    from backend.rag.embedding import SentenceTransformerEmbedder, TfidfEmbedder

    meta_path = index_dir / "meta.json"
    backend = "tfidf"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                backend = json.load(fh).get("embedder", "tfidf")
        except Exception:
            pass

    if backend == SentenceTransformerEmbedder.name:
        return SentenceTransformerEmbedder.load(index_dir)
    return TfidfEmbedder.load(index_dir)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def compose_retrieval_query(
    device_name: str,
    defect_classes: Sequence[str] = (),
    thermal_severity: str = "",
    rule_names: Sequence[str] = (),
    extra_terms: Sequence[str] = (),
) -> str:
    """由检测结果拼装检索查询串。

    把「设备 + 缺陷 + 规则名」组织成一段自然语言，
    使字符 n-gram 向量化能捕捉到与标准文本重叠的词组。
    """
    parts: List[str] = [device_name or "电力设备", "检测 异常 判断 依据 处理"]

    if defect_classes:
        parts.append("外观缺陷：" + "、".join(defect_classes))
    if thermal_severity:
        severity_zh = {
            "normal": "正常", "warning": "一般缺陷",
            "alarm": "严重缺陷", "critical": "危急缺陷",
        }.get(thermal_severity, thermal_severity)
        parts.append(f"红外测温 温升 相对温差 {severity_zh}")
    if rule_names:
        parts.append("触发规则：" + "、".join(rule_names))
    if extra_terms:
        parts.append("、".join(extra_terms))

    return " ".join(parts)


__all__ = ["KnowledgeBase", "compose_retrieval_query"]
