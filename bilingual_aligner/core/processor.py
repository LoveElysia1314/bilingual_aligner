from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

from .splitter import UniversalSplitter


try:
    import torch
except Exception:
    torch = None


class TextProcessor:
    """Text processing utilities"""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.similarity_model = None  # 延迟加载
        self.similarity_cache = {}
        self.sentence_embedding_cache = {}  # Cache for sentence-level encodings
        self.logger = logging.getLogger(__name__)

        # Determine device: prefer explicit parameter, otherwise use torch.cuda if available
        if device:
            self.device = device
        else:
            if (
                torch is not None
                and getattr(torch, "cuda", None) is not None
                and torch.cuda.is_available()
            ):
                self.device = "cuda"
            else:
                self.device = "cpu"

        # Log PyTorch / device info at INFO level
        try:
            if torch is None:
                self.logger.info(f"PyTorch not available; using device={self.device}")
            else:
                torch_version = getattr(torch, "__version__", "unknown")
                cuda_attr = getattr(torch, "cuda", None)
                cuda_available = (
                    cuda_attr is not None
                    and getattr(torch.cuda, "is_available", lambda: False)()
                )
                if cuda_available:
                    try:
                        device_count = torch.cuda.device_count()
                        device_names = [
                            torch.cuda.get_device_name(i) for i in range(device_count)
                        ]
                        names_str = (
                            ", ".join(device_names) if device_names else "<unknown>"
                        )
                        self.logger.info(
                            f"PyTorch {torch_version}; CUDA available; preferred device={self.device}; gpu_count={device_count}; gpu_names={names_str}"
                        )
                    except Exception as e:
                        self.logger.info(
                            f"PyTorch {torch_version}; CUDA available; preferred device={self.device}; failed to query GPU names: {e}"
                        )
                else:
                    self.logger.info(
                        f"PyTorch {torch_version}; CUDA not available; using device={self.device}"
                    )
        except Exception as e:
            self.logger.info(f"Unable to query PyTorch device info: {e}")

        # Hide Hugging Face related logs
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    def _load_model(self):
        """Load model on demand; prefer GPU if available, fallback to CPU on error"""
        if self.similarity_model is not None:
            return

        self.logger.info(f"正在加载语言模型 (preferred device={self.device})...")
        from sentence_transformers import SentenceTransformer

        # Try to load on preferred device; if fails, fall back to CPU
        try:
            self.similarity_model = SentenceTransformer(
                self.model_name, device=self.device, trust_remote_code=True
            )
            self.logger.info(f"模型加载完成 on {self.device}")
        except Exception as e:
            self.logger.warning(
                f"Failed to load model on {self.device}: {e}. Falling back to cpu."
            )
            try:
                self.similarity_model = SentenceTransformer(
                    self.model_name, device="cpu", trust_remote_code=True
                )
                self.device = "cpu"
                self.logger.info("模型加载完成 on cpu")
            except Exception as e2:
                self.logger.error(f"Failed to load model on cpu as well: {e2}")
                raise

    def split_sentences(self, text: str) -> List[str]:
        """Split sentences"""
        if not text.strip():
            return []

        points = UniversalSplitter._get_safe_split_points_for_counting(text)
        if not points:
            return [text.strip()]

        sentences = []
        prev = 0
        for p in points:
            sentence = text[prev:p].strip()
            if sentence:
                sentences.append(sentence)
            prev = p
        remaining = text[prev:].strip()
        if remaining:
            sentences.append(remaining)
        return sentences

    def calculate_similarity(
        self, text1: str, text2: str, method: str = "sentence"
    ) -> float:
        """Calculate similarity"""
        key = (text1, text2)
        if key in self.similarity_cache:
            return self.similarity_cache[key]

        if not text1.strip() or not text2.strip():
            similarity = 0.0
        else:
            self._load_model()  # 确保模型已加载
            if method == "paragraph":
                # Original paragraph encoding: encode both texts in one batch
                # and compute cosine similarity between the raw embeddings.
                try:
                    embeddings = self.similarity_model.encode(
                        [text1, text2], show_progress_bar=False
                    )
                    # Use sklearn cosine_similarity for numerical stability
                    similarity = float(
                        cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    )
                except Exception:
                    # Fallback: use sentence-level aggregation if paragraph encode fails
                    emb1 = self.get_normalized_embedding_by_sentences(
                        text1, method="mean"
                    )
                    emb2 = self.get_normalized_embedding_by_sentences(
                        text2, method="mean"
                    )
                    similarity = float(np.dot(emb1, emb2))
            else:
                # Default: sentence-level aggregation (mean of sentence embeddings)
                emb1 = self.get_normalized_embedding_by_sentences(text1, method="mean")
                emb2 = self.get_normalized_embedding_by_sentences(text2, method="mean")
                similarity = float(np.dot(emb1, emb2))

            # Clip to [0,1] to avoid negative artifacts in downstream code
            similarity = max(0.0, min(1.0, similarity))

        self.similarity_cache[key] = similarity
        return similarity

    def get_normalized_embedding(self, text: str) -> np.ndarray:
        """Get normalized embedding for a single text"""
        self._load_model()
        embedding = self.similarity_model.encode([text], show_progress_bar=False)[0]
        return np.array(embedding, dtype=np.float32)

    def get_normalized_embedding_by_sentences(
        self, text: str, method: str = "mean"
    ) -> np.ndarray:
        """
        Get normalized embedding by splitting text into sentences, encoding each sentence separately,
        and combining the sentence embeddings.

        Args:
            text: The text to encode
            method: How to combine sentence embeddings ('mean' for average, 'sum' for sum)

        Returns:
            Combined normalized embedding vector
        """
        if not text.strip():
            return np.zeros(
                384, dtype=np.float32
            )  # Default dimension for multilingual mpnet

        # Check cache first
        cache_key = (text, method)
        if cache_key in self.sentence_embedding_cache:
            return self.sentence_embedding_cache[cache_key]

        self._load_model()

        # Split into sentences
        sentences = self.split_sentences(text)
        if not sentences:
            return np.zeros(384, dtype=np.float32)

        # Encode all sentences in batch (convert to numpy for numeric ops)
        embeddings = self.similarity_model.encode(
            sentences, show_progress_bar=False, convert_to_numpy=True
        )

        # Normalize each sentence embedding individually to remove per-sentence
        # magnitude differences, then combine. This implements the
        # "normalize-then-combine" strategy which ensures each sentence's
        # direction matters while avoiding dominance by large-norm vectors.
        eps = 1e-8
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed_sentences = embeddings / (norms + eps)

        if method == "mean":
            combined = np.mean(normed_sentences, axis=0)
        elif method == "sum":
            combined = np.sum(normed_sentences, axis=0)
        else:
            combined = np.mean(normed_sentences, axis=0)

        # Final normalization to produce a unit vector
        result = combined / (np.linalg.norm(combined) + eps)

        # Cache the result
        self.sentence_embedding_cache[cache_key] = result
        return result

    def find_split_points(self, text: str, num_chunks: int) -> List[int]:
        """Find safe split points for dividing text into chunks"""
        return UniversalSplitter.find_split_points(text, num_chunks)

    def find_hard_split_points(self, text: str) -> List[int]:
        """Find hard split points (sentence-ending punctuation only)"""
        return UniversalSplitter.find_hard_split_points(text)

    def find_soft_split_points(self, text: str) -> List[int]:
        """Find soft split points (commas, colons, etc)"""
        return UniversalSplitter.find_soft_split_points(text)


# Simple factory / cache to reuse TextProcessor instances and avoid re-loading models
_PROCESSOR_CACHE = {}


def get_text_processor(
    model_name: Optional[str] = None, device: Optional[str] = None
) -> TextProcessor:
    """Return a cached TextProcessor for the given (model_name, device).

    This avoids loading the same large sentence-transformers model multiple times
    within the same process.
    """
    key: Tuple[Optional[str], Optional[str]] = (model_name, device)
    if key in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[key]

    tp = TextProcessor(
        model_name=model_name or "Alibaba-NLP/gte-multilingual-base",
        device=device,
    )
    _PROCESSOR_CACHE[key] = tp
    return tp
