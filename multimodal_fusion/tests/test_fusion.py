"""Tests for multimodal fusion module."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from multimodal_fusion.config import MultimodalConfig
from multimodal_fusion.schemas import (
    ConsistencyResult,
    ConsistencyStatus,
    Evidence,
    FusionVerdict,
    MultimodalResult,
)
from multimodal_fusion.consistency_checker import ConsistencyChecker
from multimodal_fusion.evidence_collector import EvidenceCollector
from multimodal_fusion.fusion_engine import FusionEngine
from multimodal_fusion.multimodal_analyzer import (
    MultimodalAnalyzer,
    get_multimodal_analyzer,
)


def test_config_initialization():
    print("Testing config initialization...")
    config = MultimodalConfig()
    assert config.sync_threshold == 0.7
    assert config.video_weight == 0.35
    print("  [PASS] Config initialized")


def test_consistency_checker():
    print("Testing consistency checker...")
    config = MultimodalConfig()
    checker = ConsistencyChecker(config)

    audio_feat = np.random.randn(10, 128)
    video_feat = np.random.randn(10, 128)

    result = checker.check_sync(audio_feat, video_feat)

    assert isinstance(result, ConsistencyResult)
    assert 0 <= result.sync_score <= 1
    print(f"  [PASS] Sync score: {result.sync_score:.3f}")


def test_consistency_checker_empty():
    print("Testing consistency checker empty...")
    config = MultimodalConfig()
    checker = ConsistencyChecker(config)

    result = checker.check_sync(np.array([]), np.array([]))

    assert result.status == ConsistencyStatus.UNKNOWN
    print("  [PASS] Empty handled correctly")


def test_lip_sync():
    print("Testing lip sync...")
    config = MultimodalConfig()
    checker = ConsistencyChecker(config)

    audio_energy = np.random.rand(20)
    mouth_movement = np.random.rand(20)

    result = checker.check_lip_sync(audio_energy, mouth_movement)

    assert isinstance(result, ConsistencyResult)
    print(f"  [PASS] Lip sync score: {result.sync_score:.3f}")


def test_evidence_collector():
    print("Testing evidence collector...")
    config = MultimodalConfig()
    collector = EvidenceCollector(config)

    evidence = collector.collect_all(
        video_score=0.3,
        audio_score=0.7,
        sync_score=0.5,
    )

    assert len(evidence) == 3
    assert all(isinstance(e, Evidence) for e in evidence)
    print(f"  [PASS] Collected {len(evidence)} evidence pieces")


def test_evidence_collector_individual():
    print("Testing evidence collector individual...")
    config = MultimodalConfig()
    collector = EvidenceCollector(config)

    video_ev = collector.collect_video_evidence(0.8)
    audio_ev = collector.collect_audio_evidence(0.6)
    sync_ev = collector.collect_sync_evidence(0.4)

    assert video_ev.source == "video_analysis"
    assert audio_ev.source == "audio_analysis"
    assert sync_ev.source == "sync_analysis"
    print("  [PASS] Individual evidence collection works")


def test_fusion_engine_weighted():
    print("Testing fusion engine weighted...")
    config = MultimodalConfig()
    engine = FusionEngine(config)

    evidence = [
        Evidence(source="video_analysis", score=0.3),
        Evidence(source="audio_analysis", score=0.8),
        Evidence(source="sync_analysis", score=0.5),
    ]

    result = engine.fuse(evidence)

    assert "verdict" in result
    assert "confidence" in result
    assert isinstance(result["verdict"], FusionVerdict)
    print(f"  [PASS] Weighted fusion: {result['verdict'].value}")


def test_fusion_engine_bayesian():
    print("Testing fusion engine bayesian...")
    config = MultimodalConfig()
    engine = FusionEngine(config)

    evidence = [
        Evidence(source="video_analysis", score=0.8, confidence=0.9),
        Evidence(source="audio_analysis", score=0.7, confidence=0.8),
    ]

    result = engine.bayesian_fusion(evidence)

    assert "verdict" in result
    assert "posterior" in result
    print(f"  [PASS] Bayesian fusion: posterior={result['posterior']:.3f}")


def test_fusion_engine_ds():
    print("Testing fusion engine DS...")
    config = MultimodalConfig()
    engine = FusionEngine(config)

    evidence = [
        Evidence(source="video_analysis", score=0.7, confidence=0.8),
        Evidence(source="audio_analysis", score=0.6, confidence=0.7),
    ]

    result = engine.ds_fusion(evidence)

    assert "verdict" in result
    assert "belief_fake" in result
    print(f"  [PASS] DS fusion: belief_fake={result['belief_fake']:.3f}")


def test_fusion_empty():
    print("Testing fusion empty...")
    config = MultimodalConfig()
    engine = FusionEngine(config)

    result = engine.fuse([])

    assert result["verdict"] == FusionVerdict.UNKNOWN
    print("  [PASS] Empty fusion handled")


def test_multimodal_analyzer():
    print("Testing multimodal analyzer...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    result = analyzer.analyze(
        video_score=0.3,
        audio_score=0.7,
        sync_score=0.6,
    )

    assert isinstance(result, MultimodalResult)
    assert isinstance(result.verdict, FusionVerdict)
    print(f"  [PASS] Analysis complete: {result.verdict.value}")


def test_multimodal_analyzer_suspicious():
    print("Testing multimodal analyzer suspicious...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    result = analyzer.analyze(
        video_score=0.8,
        audio_score=0.9,
        sync_score=0.3,
    )

    assert result.verdict in [
        FusionVerdict.SUSPICIOUS,
        FusionVerdict.POTENTIALLY_MANIPULATED,
    ]
    print(f"  [PASS] Suspicious content detected: {result.verdict.value}")


def test_multimodal_analyzer_authentic():
    print("Testing multimodal analyzer authentic...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    result = analyzer.analyze(
        video_score=0.1,
        audio_score=0.1,
        sync_score=0.9,
    )

    assert result.verdict in [
        FusionVerdict.AUTHENTIC,
        FusionVerdict.POTENTIALLY_AUTHENTIC,
    ]
    print(f"  [PASS] Authentic content: {result.verdict.value}")


def test_multimodal_result_summary():
    print("Testing multimodal result summary...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    result = analyzer.analyze(video_score=0.5, audio_score=0.5)
    summary = result.get_summary()

    assert "MULTIMODAL ANALYSIS REPORT" in summary
    print("  [PASS] Summary generated")


def test_multimodal_result_to_dict():
    print("Testing multimodal result to_dict...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    result = analyzer.analyze(video_score=0.5, audio_score=0.5)
    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert "verdict" in result_dict
    print("  [PASS] Result to_dict works")


def test_multimodal_analyzer_bayesian():
    print("Testing multimodal analyzer bayesian...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    result = analyzer.analyze(
        video_score=0.6,
        audio_score=0.7,
        method="bayesian",
    )

    assert isinstance(result, MultimodalResult)
    print(f"  [PASS] Bayesian analysis: {result.verdict.value}")


def test_multimodal_analyzer_ds():
    print("Testing multimodal analyzer DS...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    result = analyzer.analyze(
        video_score=0.6,
        audio_score=0.7,
        method="ds",
    )

    assert isinstance(result, MultimodalResult)
    print(f"  [PASS] DS analysis: {result.verdict.value}")


def test_multimodal_with_features():
    print("Testing multimodal with features...")
    config = MultimodalConfig()
    analyzer = MultimodalAnalyzer(config)

    audio_feat = np.random.randn(10, 128)
    video_feat = np.random.randn(10, 128)

    result = analyzer.analyze_with_features(
        audio_feat, video_feat,
        video_score=0.5, audio_score=0.5,
    )

    assert isinstance(result, MultimodalResult)
    print("  [PASS] Feature-based analysis works")


def test_fusion_verdicts():
    print("Testing fusion verdicts...")
    # All verdict types should be usable
    verdicts = [
        FusionVerdict.AUTHENTIC,
        FusionVerdict.POTENTIALLY_AUTHENTIC,
        FusionVerdict.SUSPICIOUS,
        FusionVerdict.POTENTIALLY_MANIPULATED,
        FusionVerdict.REQUIRES_REVIEW,
        FusionVerdict.UNKNOWN,
    ]

    for v in verdicts:
        assert isinstance(v.value, str)

    print("  [PASS] All verdict types valid")


def test_factory_function():
    print("Testing factory function...")
    analyzer = get_multimodal_analyzer()

    assert isinstance(analyzer, MultimodalAnalyzer)
    print("  [PASS] Factory works")


def test_evidence_scores():
    print("Testing evidence scores...")
    config = MultimodalConfig()
    collector = EvidenceCollector(config)

    # High scores should be suspicious
    high_ev = collector.collect_video_evidence(0.9)
    low_ev = collector.collect_video_evidence(0.1)

    assert high_ev.status == "SUSPICIOUS"
    assert low_ev.status == "NORMAL"
    print("  [PASS] Evidence scoring works")


def test_consistency_status():
    print("Testing consistency status...")
    # All status types should be usable
    statuses = [
        ConsistencyStatus.CONSISTENT,
        ConsistencyStatus.POTENTIAL_MISMATCH,
        ConsistencyStatus.INCONSISTENT,
        ConsistencyStatus.UNKNOWN,
    ]

    for s in statuses:
        assert isinstance(s.value, str)

    print("  [PASS] All consistency statuses valid")


def run_all_tests():
    print("=" * 60)
    print("MULTIMODAL FUSION TESTS")
    print("=" * 60)

    tests = [
        test_config_initialization,
        test_consistency_checker,
        test_consistency_checker_empty,
        test_lip_sync,
        test_evidence_collector,
        test_evidence_collector_individual,
        test_fusion_engine_weighted,
        test_fusion_engine_bayesian,
        test_fusion_engine_ds,
        test_fusion_empty,
        test_multimodal_analyzer,
        test_multimodal_analyzer_suspicious,
        test_multimodal_analyzer_authentic,
        test_multimodal_result_summary,
        test_multimodal_result_to_dict,
        test_multimodal_analyzer_bayesian,
        test_multimodal_analyzer_ds,
        test_multimodal_with_features,
        test_fusion_verdicts,
        test_factory_function,
        test_evidence_scores,
        test_consistency_status,
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
