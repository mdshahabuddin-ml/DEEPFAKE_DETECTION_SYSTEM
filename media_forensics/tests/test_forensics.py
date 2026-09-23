"""Tests for media forensics module."""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from media_forensics.config import MediaForensicsConfig
from media_forensics.file_hasher import FileHasher
from media_forensics.format_analyzer import FormatAnalyzer
from media_forensics.integrity_checker import IntegrityChecker
from media_forensics.metadata_analyzer import MetadataAnalyzer
from media_forensics.forensic_analyzer import (
    ForensicAnalyzer,
    ForensicResult,
    get_forensic_analyzer,
)


def create_test_file(content: bytes, ext: str = ".bin") -> str:
    """Create a temporary test file."""
    temp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp.write(content)
    temp.close()
    return temp.name


def create_test_wav() -> str:
    """Create a minimal WAV file."""
    # Create minimal WAV
    sample_rate = 16000
    channels = 1
    bits = 16
    data = b"\x00\x00" * 100  # 100 samples of silence

    # WAV header
    header = b"RIFF"
    header += struct.pack("<I", 36 + len(data))
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)  # chunk size
    header += struct.pack("<H", 1)   # PCM
    header += struct.pack("<H", channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", sample_rate * channels * bits // 8)
    header += struct.pack("<H", channels * bits // 8)
    header += struct.pack("<H", bits)
    header += b"data"
    header += struct.pack("<I", len(data))

    return create_test_file(header + data, ".wav")


def create_test_png() -> str:
    """Create a minimal PNG file."""
    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    ihdr_data = struct.pack(">II", 100, 100)  # width, height
    ihdr_data += struct.pack("BBBBB", 8, 2, 0, 0, 0)  # bit depth, color type
    ihdr_crc = b"\x00" * 4  # Dummy CRC
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc

    return create_test_file(signature + ihdr, ".png")


def test_config_initialization():
    print("Testing config initialization...")
    config = MediaForensicsConfig()
    assert "sha256" in config.hash_algorithms
    assert ".mp4" in config.supported_video_formats
    print("  [PASS] Config initialized")


def test_file_hasher():
    print("Testing file hasher...")
    hasher = FileHasher(["sha256", "md5"])

    # Create test file
    test_file = create_test_file(b"test content for hashing")

    try:
        hashes = hasher.hash_file(test_file)

        assert "sha256" in hashes
        assert "md5" in hashes
        assert len(hashes["sha256"]) == 64
        assert len(hashes["md5"]) == 32
        print(f"  [PASS] SHA256: {hashes['sha256'][:16]}...")
    finally:
        os.unlink(test_file)


def test_file_hasher_bytes():
    print("Testing file hasher bytes...")
    hasher = FileHasher(["sha256"])

    hashes = hasher.hash_bytes(b"test data")

    assert "sha256" in hashes
    assert len(hashes["sha256"]) == 64
    print("  [PASS] Bytes hashing works")


def test_evidence_id():
    print("Testing evidence ID...")
    hasher = FileHasher()

    test_file = create_test_file(b"test")

    try:
        evidence_id = hasher.generate_evidence_id(test_file)

        assert evidence_id.startswith("EV-")
        assert len(evidence_id) == 19  # EV- + 16 chars
        print(f"  [PASS] Evidence ID: {evidence_id}")
    finally:
        os.unlink(test_file)


def test_hash_verification():
    print("Testing hash verification...")
    hasher = FileHasher(["sha256"])

    test_file = create_test_file(b"verify this")

    try:
        hashes = hasher.hash_file(test_file)
        correct_hash = hashes["sha256"]

        # Verify correct hash
        assert hasher.verify_integrity(test_file, correct_hash, "sha256")

        # Verify wrong hash
        assert not hasher.verify_integrity(test_file, "wrong_hash", "sha256")

        print("  [PASS] Hash verification works")
    finally:
        os.unlink(test_file)


def test_format_analyzer():
    print("Testing format analyzer...")
    analyzer = FormatAnalyzer()

    test_file = create_test_wav()

    try:
        result = analyzer.analyze(test_file)

        assert "extension" in result
        assert result["extension"] == ".wav"
        assert result["media_type"] == "audio"
        print(f"  [PASS] Format: {result['media_type']}")
    finally:
        os.unlink(test_file)


def test_format_analyzer_video():
    print("Testing format analyzer video...")
    analyzer = FormatAnalyzer()

    # Test extension detection
    assert analyzer._detect_media_type(".mp4") == "video"
    assert analyzer._detect_media_type(".mp3") == "audio"
    assert analyzer._detect_media_type(".jpg") == "image"
    assert analyzer._detect_media_type(".xyz") == "unknown"

    print("  [PASS] Media type detection works")


def test_integrity_checker():
    print("Testing integrity checker...")
    checker = IntegrityChecker()

    test_file = create_test_wav()

    try:
        result = checker.check_file(test_file)

        assert "valid" in result
        assert "checks" in result
        assert len(result["checks"]) > 0
        print(f"  [PASS] Integrity valid: {result['valid']}")
    finally:
        os.unlink(test_file)


def test_integrity_checker_nonexistent():
    print("Testing integrity checker nonexistent...")
    checker = IntegrityChecker()

    result = checker.check_file("nonexistent.txt")

    assert result["valid"] == False
    assert "error" in result
    print("  [PASS] Nonexistent file handled")


def test_integrity_compression():
    print("Testing integrity compression detection...")
    checker = IntegrityChecker()

    result = checker.detect_compression_artifacts("test.mp4")

    assert "is_compressed" in result
    assert result["is_compressed"] == True

    result2 = checker.detect_compression_artifacts("test.wav")
    assert result2["is_compressed"] == False

    print("  [PASS] Compression detection works")


def test_metadata_analyzer():
    print("Testing metadata analyzer...")
    analyzer = MetadataAnalyzer()

    test_file = create_test_wav()

    try:
        result = analyzer.analyze(test_file)

        assert "file_name" in result
        assert "file_size" in result
        assert "format" in result
        print(f"  [PASS] Metadata extracted")
    finally:
        os.unlink(test_file)


def test_metadata_analyzer_png():
    print("Testing metadata analyzer PNG...")
    analyzer = MetadataAnalyzer()

    test_file = create_test_png()

    try:
        result = analyzer.analyze(test_file)

        assert "format" in result
        assert result["format"] == "png"
        print(f"  [PASS] PNG metadata extracted")
    finally:
        os.unlink(test_file)


def test_metadata_sanitization():
    print("Testing metadata sanitization...")
    analyzer = MetadataAnalyzer(sanitize=True)

    # Test sanitization
    metadata = {
        "format": "mp4",
        "gps_data": "sensitive",
        "camera_model": "iPhone",
        "width": 1920,
    }

    sanitized = analyzer._sanitize_metadata(metadata)

    assert "format" in sanitized
    assert "width" in sanitized
    assert "gps_data" not in sanitized
    print("  [PASS] Metadata sanitized")


def test_metadata_manipulation_indicators():
    print("Testing metadata manipulation indicators...")
    analyzer = MetadataAnalyzer()

    metadata = {"has_exif": True, "parse_error": None}
    result = analyzer.detect_manipulation_indicators(metadata)

    assert "indicators" in result
    assert result["n_indicators"] > 0
    print(f"  [PASS] Found {result['n_indicators']} indicators")


def test_forensic_analyzer():
    print("Testing forensic analyzer...")
    config = MediaForensicsConfig()
    analyzer = ForensicAnalyzer(config)

    test_file = create_test_wav()

    try:
        result = analyzer.analyze(test_file)

        assert isinstance(result, ForensicResult)
        assert result.evidence_id.startswith("EV-")
        assert "sha256" in result.hashes
        assert result.is_valid
        print(f"  [PASS] Analysis complete: {result.evidence_id}")
    finally:
        os.unlink(test_file)


def test_forensic_result_summary():
    print("Testing forensic result summary...")
    config = MediaForensicsConfig()
    analyzer = ForensicAnalyzer(config)

    test_file = create_test_wav()

    try:
        result = analyzer.analyze(test_file)
        summary = result.get_summary()

        assert "MEDIA FORENSIC ANALYSIS REPORT" in summary
        assert result.evidence_id in summary
        print("  [PASS] Summary generated")
    finally:
        os.unlink(test_file)


def test_forensic_result_to_dict():
    print("Testing forensic result to_dict...")
    config = MediaForensicsConfig()
    analyzer = ForensicAnalyzer(config)

    test_file = create_test_wav()

    try:
        result = analyzer.analyze(test_file)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "evidence_id" in result_dict
        assert "hashes" in result_dict
        print("  [PASS] Result to_dict works")
    finally:
        os.unlink(test_file)


def test_forensic_hash_verification():
    print("Testing forensic hash verification...")
    config = MediaForensicsConfig()
    analyzer = ForensicAnalyzer(config)

    test_file = create_test_wav()

    try:
        # Get correct hash
        result = analyzer.analyze(test_file)
        correct_hash = result.hashes["sha256"]

        # Verify
        assert analyzer.verify_hash(test_file, correct_hash, "sha256")
        assert not analyzer.verify_hash(test_file, "wrong", "sha256")
        print("  [PASS] Hash verification works")
    finally:
        os.unlink(test_file)


def test_quick_info():
    print("Testing quick info...")
    config = MediaForensicsConfig()
    analyzer = ForensicAnalyzer(config)

    test_file = create_test_wav()

    try:
        info = analyzer.get_quick_info(test_file)

        assert "media_type" in info
        print(f"  [PASS] Quick info: {info['media_type']}")
    finally:
        os.unlink(test_file)


def test_factory_function():
    print("Testing factory function...")
    analyzer = get_forensic_analyzer()

    assert isinstance(analyzer, ForensicAnalyzer)
    print("  [PASS] Factory works")


def run_all_tests():
    print("=" * 60)
    print("MEDIA FORENSICS TESTS")
    print("=" * 60)

    tests = [
        test_config_initialization,
        test_file_hasher,
        test_file_hasher_bytes,
        test_evidence_id,
        test_hash_verification,
        test_format_analyzer,
        test_format_analyzer_video,
        test_integrity_checker,
        test_integrity_checker_nonexistent,
        test_integrity_compression,
        test_metadata_analyzer,
        test_metadata_analyzer_png,
        test_metadata_sanitization,
        test_metadata_manipulation_indicators,
        test_forensic_analyzer,
        test_forensic_result_summary,
        test_forensic_result_to_dict,
        test_forensic_hash_verification,
        test_quick_info,
        test_factory_function,
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
