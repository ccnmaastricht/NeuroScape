from dataclasses import dataclass
from typing import List


@dataclass
class Article:
    """Data class for an article.
    """
    pmid: int
    doi: str
    title: str
    type: str
    journal: str
    year: int
    age: float
    citation_count: int
    citation_rate: float
    abstract: str
    embedding: List[float]
    in_links: List[int]
    out_links: List[int]


@dataclass
class Embeddings:
    """Data class for an embedding.
    """
    pmids: List[int]
    embeddings: List[float]


@dataclass
class EmbeddingsWithLabels:
    """Data class for an embedding with labels.
    """
    pmids: List[int]
    embeddings: List[float]
    labels: List[int]
