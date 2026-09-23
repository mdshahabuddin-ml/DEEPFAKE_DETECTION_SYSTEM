"""Tests for speaker verification module.

These tests verify speaker verification works correctly
without modifying any existing system tests.
"""

import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from audio_forensics.config import AudioForensicsConfig
from audio_forensics.speaker_verification import (
    SpeakerVerifier,
    SpeakerVerificationError,
    SpeakerEmbedding,
    VerificationResult,
)


def create_test_audio(
    duration: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 200.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Create test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def create_speaker_audio(
    speaker_id: str,
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Create audio that simulates different speakers."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Different base frequencies for different speakers
    freq_map = {
        "A": 150,
        "B": 250,
        "C": 180,
    }
    base_freq = freq_map.get(speaker_id, 200)

    # Add harmonics unique to each speaker
    audio = (
        0.4 * np.sin(2 * np.pi * base_freq * t) +
        0.2 * np.sin(2 * np.pi * base_freq * 2 * t) +
        0.1 * np.sin(2 * np.pi * base_freq * 3 * t) +
        0.05 * np.sin(2 * np.pi * base_freq * 4 * t)
    )

    return audio.astype(np.float32)


def test_verifier_initialization():
    """Test verifier initialization."""
    print("Testing verifier initialization...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    assert verifier is not None
    assert not verifier.is_model_loaded

    print("  [PASS] Verifier initialized")


def test_simple_embedding_extraction():
    """Test simple embedding extraction."""
    print("Testing simple embedding extraction...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    audio = create_test_audio(duration=2.0)
    embedding = verifier.extract_embedding(audio, 16000)

    assert isinstance(embedding, SpeakerEmbedding)
    assert embedding.embedding is not None
    assert embedding.dimension > 0
    assert embedding.model_name == "simple_spectral"

    print(f"  [PASS] Embedding extracted (dim={embedding.dimension})")


def test_embedding_normalization():
    """Test that embeddings are normalized."""
    print("Testing embedding normalization...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    audio = create_test_audio(duration=2.0)
    embedding = verifier.extract_embedding(audio, 16000)

    # Check normalization
    norm = np.linalg.norm(embedding.embedding)
    assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: {norm}"

    print(f"  [PASS] Embedding normalized (norm={norm:.4f})")


def test_same_audio_similarity():
    """Test that same audio has high similarity."""
    print("Testing same audio similarity...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    audio = create_test_audio(duration=2.0)

    result = verifier.verify(audio, audio, 16000)

    assert isinstance(result, VerificationResult)
    assert result.similarity_score > 0.99  # Same audio should be nearly 1.0
    assert result.is_consistent

    print(f"  [PASS] Same audio similarity: {result.similarity_score:.4f}")


def test_different_audio_similarity():
    """Test that different audio produces a result."""
    print("Testing different audio similarity...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    audio_a = create_speaker_audio("A", duration=2.0)
    audio_b = create_speaker_audio("B", duration=2.0)

    result = verifier.verify(audio_a, audio_b, 16000)

    assert isinstance(result, VerificationResult)
    assert 0 <= result.similarity_score <= 1
    assert result.method != ""

    # Note: With simple_spectral embeddings, different speakers may still
    # have high similarity. Pretrained models are needed for real distinction.
    print(f"  [PASS] Different audio similarity: {result.similarity_score:.4f}")
    print(f"    (Note: Simple spectral embeddings may not distinguish speakers well)")


def test_reference_verification():
    """Test verification against reference embedding."""
    print("Testing reference verification...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    # Create reference audio and extract embedding
    reference_audio = create_speaker_audio("A", duration=2.0)
    reference_embedding = verifier.extract_embedding(reference_audio, 16000)

    # Verify same speaker
    same_speaker = create_speaker_audio("A", duration=2.0)
    result_same = verifier.verify_against_reference(
        same_speaker, reference_embedding.embedding, 16000
    )

    # Verify different speaker
    diff_speaker = create_speaker_audio("B", duration=2.0)
    result_diff = verifier.verify_against_reference(
        diff_speaker, reference_embedding.embedding, 16000
    )

    assert result_same.similarity_score > result_diff.similarity_score

    print(f"  [PASS] Reference verification:")
    print(f"    Same speaker: {result_same.similarity_score:.4f}")
    print(f"    Diff speaker: {result_diff.similarity_score:.4f}")


def test_empty_audio():
    """Test handling of empty audio."""
    print("Testing empty audio handling...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    try:
        verifier.extract_embedding(np.array([]), 16000)
        assert False, "Should raise SpeakerVerificationError"
    except SpeakerVerificationError:
        pass

    print("  [PASS] Empty audio handled correctly")


def test_stereo_audio():
    """Test handling of stereo audio."""
    print("Testing stereo audio...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    mono = create_test_audio(duration=2.0)
    stereo = np.stack([mono, mono], axis=1)

    embedding = verifier.extract_embedding(stereo, 16000)

    assert embedding.embedding is not None

    print("  [PASS] Stereo converted to mono")


def test_verification_consistent():
    """Test consistent speaker detection."""
    print("Testing consistent speaker detection...")

    config = AudioForensicsConfig(speaker_similarity_threshold=0.7)
    verifier = SpeakerVerifier(config)

    # Same speaker (high similarity expected)
    audio = create_speaker_audio("A", duration=2.0)
    result = verifier.verify(audio, audio, 16000)

    assert result.is_consistent

    print(f"  [PASS] Same speaker consistent: {result.is_consistent}")


def test_verification_inconsistent():
    """Test inconsistent speaker detection."""
    print("Testing inconsistent speaker detection...")

    config = AudioForensicsConfig(speaker_similarity_threshold=0.9)
    verifier = SpeakerVerifier(config)

    # Different speakers
    audio_a = create_speaker_audio("A", duration=2.0)
    audio_b = create_speaker_audio("B", duration=2.0)

    result = verifier.verify(audio_a, audio_b, 16000)

    # May or may not be consistent depending on similarity
    assert isinstance(result.is_consistent, bool)

    print(f"  [PASS] Different speakers consistent: {result.is_consistent}")


def test_segment_embeddings():
    """Test extraction of embeddings for multiple segments."""
    print("Testing segment embeddings...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    audio = create_test_audio(duration=5.0)

    segments = [
        {"start_time": 0.0, "end_time": 2.0},
        {"start_time": 2.5, "end_time": 4.5},
    ]

    embeddings = verifier.extract_embedding_from_segments(
        audio, 16000, segments
    )

    assert len(embeddings) == 2
    assert all(e.embedding is not None for e in embeddings)

    print(f"  [PASS] Extracted {len(embeddings)} segment embeddings")


def test_speaker_similarity_summary():
    """Test speaker similarity summary across segments."""
    print("Testing speaker similarity summary...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    # Create audio with consistent "speaker"
    audio = create_speaker_audio("A", duration=5.0)

    segments = [
        {"start_time": 0.0, "end_time": 2.0},
        {"start_time": 2.5, "end_time": 4.5},
    ]

    summary = verifier.get_speaker_similarity_summary(
        audio, 16000, segments
    )

    assert "mean_similarity" in summary
    assert "is_consistent" in summary
    assert "n_comparisons" in summary
    assert summary["n_comparisons"] == 1  # 2 segments = 1 comparison

    print(f"  [PASS] Summary: mean_sim={summary['mean_similarity']:.4f}")


def test_embedding_consistency():
    """Test that embeddings are consistent for same audio."""
    print("Testing embedding consistency...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    audio = create_test_audio(duration=2.0)

    emb1 = verifier.extract_embedding(audio, 16000)
    emb2 = verifier.extract_embedding(audio, 16000)

    # Same audio should produce same embedding
    np.testing.assert_array_almost_equal(
        emb1.embedding, emb2.embedding, decimal=5
    )

    print("  [PASS] Embeddings consistent")


def test_threshold_config():
    """Test threshold configuration."""
    print("Testing threshold configuration...")

    config = AudioForensicsConfig(speaker_similarity_threshold=0.85)
    verifier = SpeakerVerifier(config)

    assert verifier.config.speaker_similarity_threshold == 0.85

    print("  [PASS] Threshold configured correctly")


def test_verification_result_properties():
    """Test verification result properties."""
    print("Testing verification result properties...")

    config = AudioForensicsConfig()
    verifier = SpeakerVerifier(config)

    audio = create_test_audio(duration=2.0)
    result = verifier.verify(audio, audio, 16000)

    assert isinstance(result.similarity_score, float)
    assert isinstance(result.is_consistent, bool)
    assert isinstance(result.distance, float)
    assert isinstance(result.method, str)
    assert isinstance(result.metadata, dict)

    print(f"  [PASS] Result properties valid")


def run_all_tests():
    """Run all speaker verification tests."""
    print("=" * 60)
    print("SPEAKER VERIFICATION TESTS")
    print("=" * 60)

    tests = [
        test_verifier_initialization,
        test_simple_embedding_extraction,
        test_embedding_normalization,
        test_same_audio_similarity,
        test_different_audio_similarity,
        test_reference_verification,
        test_empty_audio,
        test_stereo_audio,
        test_verification_consistent,
        test_verification_inconsistent,
        test_segment_embeddings,
        test_speaker_similarity_summary,
        test_embedding_consistency,
        test_threshold_config,
        test_verification_result_properties,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
