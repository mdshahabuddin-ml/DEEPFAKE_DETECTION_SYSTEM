"""Speaker verification for audio forensic analysis.

IMPORTANT CONCEPT:
    Speaker identity and audio authenticity are NOT the same thing.

    Speaker verification answers:
        "Does this voice resemble the claimed speaker?"

    Audio deepfake detection answers:
        "Does this audio contain characteristics associated with
         synthetic or manipulated speech?"

    Do not allow speaker similarity to automatically override the
    audio deepfake detector.
    Do not automatically classify audio as authentic because the
    speaker matches.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import AudioForensicsConfig

logger = logging.getLogger(__name__)


class SpeakerVerificationError(Exception):
    """Custom exception for speaker verification errors."""
    pass


@dataclass
class SpeakerEmbedding:
    """Speaker embedding vector."""
    embedding: np.ndarray = None
    speaker_id: str = ""
    model_name: str = ""
    dimension: int = 0

    def __post_init__(self):
        if self.embedding is not None and self.dimension == 0:
            self.dimension = len(self.embedding)


@dataclass
class VerificationResult:
    """Result of speaker verification."""
    similarity_score: float = 0.0
    is_consistent: bool = False
    distance: float = 0.0
    method: str = ""
    embedding_a: Optional[SpeakerEmbedding] = None
    embedding_b: Optional[SpeakerEmbedding] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpeakerVerifier:
    """Speaker verification for audio forensics.

    This class provides speaker identity verification capabilities.
    It is completely separate from deepfake detection.

    Usage:
        config = AudioForensicsConfig()
        verifier = SpeakerVerifier(config)
        result = verifier.verify(audio_a, audio_b, sample_rate)
    """

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        self.config = config or AudioForensicsConfig()
        self._model = None
        self._model_loaded = False
        self._embedding_model = None

    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded

    def load_model(self, model_path: Optional[str] = None) -> bool:
        try:
            try:
                from speechbrain.inference import EncoderClassifier
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                )
                self._model_loaded = True
                logger.info("Loaded SpeechBrain ECAPA model")
                return True
            except ImportError:
                pass

            try:
                from resemblyzer import VoiceEncoder
                self._embedding_model = VoiceEncoder()
                self._model_loaded = True
                logger.info("Loaded Resemblyzer model")
                return True
            except ImportError:
                pass

            logger.warning("No speaker verification model available")
            return False

        except Exception as e:
            logger.error(f"Failed to load speaker model: {e}")
            return False

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int,
    ) -> SpeakerEmbedding:
        if audio is None or len(audio) == 0:
            raise SpeakerVerificationError("Empty audio")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        try:
            if self._model is not None:
                return self._extract_with_model(audio, sample_rate)
            elif self._embedding_model is not None:
                return self._extract_with_resemblyzer(audio, sample_rate)
            else:
                return self._extract_simple_embedding(audio, sample_rate)
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return self._extract_simple_embedding(audio, sample_rate)

    def _extract_with_model(
        self, audio: np.ndarray, sample_rate: int,
    ) -> SpeakerEmbedding:
        import torch
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        with torch.no_grad():
            embedding = self._model.encode_batch(audio_tensor)
        embedding = embedding.squeeze().numpy()
        return SpeakerEmbedding(
            embedding=embedding, model_name="pretrained",
            dimension=len(embedding),
        )

    def _extract_with_resemblyzer(
        self, audio: np.ndarray, sample_rate: int,
    ) -> SpeakerEmbedding:
        if sample_rate != 16000:
            from scipy.signal import resample
            n_samples = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, n_samples)
        embedding = self._embedding_model.embed_utterance(audio)
        return SpeakerEmbedding(
            embedding=embedding, model_name="resemblyzer",
            dimension=len(embedding),
        )

    def _extract_simple_embedding(
        self, audio: np.ndarray, sample_rate: int,
    ) -> SpeakerEmbedding:
        try:
            from scipy.signal import stft
            f, t, Zxx = stft(audio, fs=sample_rate, nperseg=2048, noverlap=1536)
            magnitude = np.abs(Zxx)
        except ImportError:
            magnitude = np.array([
                np.mean(audio ** 2), np.std(audio),
                np.max(np.abs(audio)), np.min(audio),
            ])

        if magnitude.ndim > 1:
            stats = np.concatenate([
                np.mean(magnitude, axis=1)[:10],
                np.std(magnitude, axis=1)[:10],
            ])
        else:
            stats = magnitude[:20] if len(magnitude) >= 20 else np.pad(magnitude, (0, 20 - len(magnitude)))

        embedding = np.concatenate([
            [np.mean(magnitude)], [np.std(magnitude)],
            [np.max(magnitude)], [np.min(magnitude)],
            stats,
        ])

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return SpeakerEmbedding(
            embedding=embedding, model_name="simple_spectral",
            dimension=len(embedding),
        )

    def verify(
        self, audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int,
    ) -> VerificationResult:
        try:
            embedding_a = self.extract_embedding(audio_a, sample_rate)
            embedding_b = self.extract_embedding(audio_b, sample_rate)
            return self._compare_embeddings(embedding_a, embedding_b)
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                similarity_score=0.0, is_consistent=False,
                method="failed", metadata={"error": str(e)},
            )

    def verify_against_reference(
        self, audio: np.ndarray, reference_embedding: np.ndarray, sample_rate: int,
    ) -> VerificationResult:
        try:
            embedding = self.extract_embedding(audio, sample_rate)
            ref_embedding = SpeakerEmbedding(
                embedding=reference_embedding, model_name="reference",
                dimension=len(reference_embedding),
            )
            return self._compare_embeddings(embedding, ref_embedding)
        except Exception as e:
            logger.error(f"Reference verification failed: {e}")
            return VerificationResult(
                similarity_score=0.0, is_consistent=False,
                method="failed", metadata={"error": str(e)},
            )

    def _compare_embeddings(
        self, embedding_a: SpeakerEmbedding, embedding_b: SpeakerEmbedding,
    ) -> VerificationResult:
        if embedding_a.embedding is None or embedding_b.embedding is None:
            return VerificationResult(
                similarity_score=0.0, is_consistent=False, method="no_embedding",
            )

        min_dim = min(len(embedding_a.embedding), len(embedding_b.embedding))
        emb_a = embedding_a.embedding[:min_dim]
        emb_b = embedding_b.embedding[:min_dim]

        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

        distance = float(np.linalg.norm(emb_a - emb_b))

        is_consistent = similarity >= self.config.speaker_similarity_threshold

        return VerificationResult(
            similarity_score=similarity,
            is_consistent=is_consistent,
            distance=distance,
            method=f"{embedding_a.model_name}_vs_{embedding_b.model_name}",
            embedding_a=embedding_a,
            embedding_b=embedding_b,
            metadata={
                "threshold": self.config.speaker_similarity_threshold,
                "dimension": min_dim,
            },
        )

    def extract_embedding_from_segments(
        self, audio: np.ndarray, sample_rate: int,
        segments: List[Dict[str, Any]],
    ) -> List[SpeakerEmbedding]:
        embeddings = []
        for seg in segments:
            start_sample = int(seg["start_time"] * sample_rate)
            end_sample = int(seg["end_time"] * sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            segment_audio = audio[start_sample:end_sample]
            if len(segment_audio) > 0:
                embedding = self.extract_embedding(segment_audio, sample_rate)
                embeddings.append(embedding)
            else:
                embeddings.append(SpeakerEmbedding())
        return embeddings

    def get_speaker_similarity_summary(
        self, audio: np.ndarray, sample_rate: int,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        embeddings = self.extract_embedding_from_segments(
            audio, sample_rate, segments
        )

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if (embeddings[i].embedding is not None and
                        embeddings[j].embedding is not None):
                    result = self._compare_embeddings(embeddings[i], embeddings[j])
                    similarities.append(result.similarity_score)

        if not similarities:
            return {
                "mean_similarity": 0.0,
                "std_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "is_consistent": False,
                "n_comparisons": 0,
            }

        sims = np.array(similarities)
        mean_sim = float(np.mean(sims))
        is_consistent = mean_sim >= self.config.speaker_similarity_threshold

        return {
            "mean_similarity": mean_sim,
            "std_similarity": float(np.std(sims)),
            "min_similarity": float(np.min(sims)),
            "max_similarity": float(np.max(sims)),
            "is_consistent": is_consistent,
            "n_comparisons": len(similarities),
        }


def get_speaker_verifier(
    config: Optional[AudioForensicsConfig] = None,
) -> SpeakerVerifier:
    return SpeakerVerifier(config)
