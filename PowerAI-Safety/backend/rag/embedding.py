"""文本向量化。

默认使用 **TF-IDF 字符 n-gram**：

    * 完全离线，无需下载模型权重，适合本机（无 GPU）环境；
    * 字符 n-gram 对中文无需分词即可工作，
      例「变压器过热」会被拆成「变压」「压器」「器过」… 的二元/三元组；
    * 配置 ``tokenizer: jieba`` 时改用 jieba 词级切分，检索语义更准。

可选接入 ``sentence-transformers`` 做语义向量；未安装时自动回退 TF-IDF，
并在日志中说明。此处不做向量维度混合，回退是整体替换而非部分降级。
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class BaseEmbedder(ABC):
    """向量化接口。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "BaseEmbedder":
        """在语料上拟合。"""

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """编码为「行向量已 L2 归一化」的矩阵，形状 ``(N, dim)``。

        归一化后余弦相似度等于内积，检索时只需一次矩阵乘法。
        """

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @abstractmethod
    def save(self, directory: PathLike) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, directory: PathLike) -> "BaseEmbedder":
        ...


class TfidfEmbedder(BaseEmbedder):
    """基于 scikit-learn 的 TF-IDF 向量化。"""

    name = "tfidf"

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 3),
        tokenizer: str = "char",
        max_features: int = 60000,
        min_df: int = 1,
    ) -> None:
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.tokenizer = tokenizer
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self._vectorizer = None
        self.dim = 0

    def _build_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.tokenizer == "jieba":
            try:
                import jieba

                def analyzer(text: str) -> List[str]:
                    return [token for token in jieba.lcut(str(text)) if token.strip()]

                return TfidfVectorizer(
                    analyzer=analyzer,
                    max_features=self.max_features,
                    min_df=self.min_df,
                    sublinear_tf=True,
                    norm="l2",
                )
            except ImportError:
                logger.warning("未安装 jieba，自动回退字符 n-gram 分词")

        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
            norm="l2",
        )

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        texts = [str(t) for t in texts if str(t).strip()]
        if not texts:
            raise ValueError("TfidfEmbedder.fit 收到空语料")
        self._vectorizer = self._build_vectorizer()
        self._vectorizer.fit(texts)
        self.dim = len(self._vectorizer.vocabulary_)
        logger.info("TF-IDF 拟合完成：%d 篇文档，%d 维特征", len(texts), self.dim)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("Embedder 尚未 fit")
        matrix = self._vectorizer.transform([str(t) for t in texts])
        # 语料规模为知识库级别（数百到数千块），稠密化内存开销可接受，
        # 换来检索时纯粹的一次矩阵乘法，无稀疏矩阵乘法开销。
        return np.asarray(matrix.todense(), dtype=np.float32)

    def save(self, directory: PathLike) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "ngram_range": self.ngram_range,
            "tokenizer": self.tokenizer,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "dim": self.dim,
            "vectorizer": self._vectorizer,
        }
        # 注意：pickle 仅用于加载本系统自己生成的索引文件，不要加载外部来源的
        # .pkl 文件——反序列化不可信内容可执行任意代码。
        with (directory / "embedder.pkl").open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, directory: PathLike) -> "TfidfEmbedder":
        path = Path(directory) / "embedder.pkl"
        if not path.exists():
            raise FileNotFoundError(f"向量化器文件不存在：{path}")
        with path.open("rb") as fh:
            payload = pickle.load(fh)

        embedder = cls(
            ngram_range=payload.get("ngram_range", (2, 3)),
            tokenizer=payload.get("tokenizer", "char"),
            max_features=payload.get("max_features", 60000),
            min_df=payload.get("min_df", 1),
        )
        embedder._vectorizer = payload.get("vectorizer")
        embedder.dim = int(payload.get("dim", 0))
        return embedder

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None


class SentenceTransformerEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的语义向量（可选，需额外安装）。"""

    name = "sentence_transformers"

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 0

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "未安装 sentence-transformers。安装：pip install sentence-transformers；"
                "或把 config.yaml 中 rag.embedding.backend 改为 tfidf。"
            ) from exc
        logger.info("加载语义向量模型：%s（首次使用需下载权重）", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        # 预训练语义模型无需拟合，此方法仅为满足接口
        self._ensure_model()
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(
            [str(t) for t in texts],
            normalize_embeddings=True,   # 与 TF-IDF 输出口径一致：L2 归一化
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def save(self, directory: PathLike) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "embedder.pkl").open("wb") as fh:
            pickle.dump({"model_name": self.model_name, "dim": self.dim}, fh)

    @classmethod
    def load(cls, directory: PathLike) -> "SentenceTransformerEmbedder":
        path = Path(directory) / "embedder.pkl"
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        embedder = cls(model_name=payload.get("model_name", "shibing624/text2vec-base-chinese"))
        embedder.dim = int(payload.get("dim", 0))
        return embedder


def create_embedder(config, force_tfidf: bool = False) -> BaseEmbedder:
    """按配置创建向量化器，缺失依赖时回退 TF-IDF。"""
    cfg = config.rag.embedding
    backend = str(cfg.get("backend", "tfidf")).lower()

    if backend == "sentence_transformers" and not force_tfidf:
        try:
            embedder = SentenceTransformerEmbedder(str(cfg.get("model", "shibing624/text2vec-base-chinese")))
            embedder._ensure_model()
            return embedder
        except Exception as exc:
            logger.warning("语义向量后端不可用（%s），回退 TF-IDF", exc)

    return TfidfEmbedder(
        ngram_range=(int(cfg.get("ngram_min", 2)), int(cfg.get("ngram_max", 3))),
        tokenizer=str(cfg.get("tokenizer", "char")),
    )


__all__ = [
    "BaseEmbedder",
    "TfidfEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
]
