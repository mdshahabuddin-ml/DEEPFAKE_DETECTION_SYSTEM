"""
Test script for forensic report generation.
Creates a sample forensic report to verify the system works.
"""

import os
import tempfile
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.forensic_report_builder import ForensicReportBuilder


def test_forensic_report_generation():
    """Test forensic report generation with sample data."""
    
    # Create sample analysis results
    visual_result = {
        "faces_detected": 1,
        "face_consistency": 0.108,
        "temporal_consistency": 0.234,
        "texture_score": 0.156,
        "fake_probability": 0.892,
        "manipulated_frames": [12, 23, 34, 45, 56, 67, 78, 89, 100, 111, 122, 133],
        "frame_analysis": [
            {"frame_idx": 0, "fake_prob": 0.12, "consistency": 0.89},
            {"frame_idx": 30, "fake_prob": 0.78, "consistency": 0.23},
            {"frame_idx": 60, "fake_prob": 0.92, "consistency": 0.15},
            {"frame_idx": 90, "fake_prob": 0.85, "consistency": 0.18},
            {"frame_idx": 120, "fake_prob": 0.91, "consistency": 0.12},
        ],
        "uncertainty": 0.08,
    }
    
    audio_result = {
        "fake_probability": 0.823,
        "quality_score": 0.678,
        "snr_estimate": 32.5,
        "spectral_centroid": 1850.2,
        "clipping_ratio": 0.0012,
        "duration": 6.45,
        "uncertainty": 0.12,
    }
    
    speaker_result = {
        "claimed_speaker": "John Doe",
        "similarity": 0.456,
        "is_verified": False,
        "confidence": 0.789,
        "enrollment_samples": 3,
    }
    
    sync_result = {
        "sync_score": 0.312,
        "anomalies_count": 7,
        "anomalies": [
            {"frame_idx": 45, "type": "desynchronization", "severity": 0.8},
            {"frame_idx": 67, "type": "audio_without_visual", "severity": 0.6},
            {"frame_idx": 89, "type": "visual_without_audio", "severity": 0.5},
        ],
        "consistency": 0.289,
        "timeline": [],
    }
    
    provenance_result = {
        "status": "not_found",
        "trust_level": "none",
        "has_c2pa": False,
        "metadata_consistent": True,
        "editing_indicators": {},
    }
    
    # Create a temporary file to simulate media file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(b"sample video data")
        tmp_path = tmp.name
    
    try:
        # Build report
        builder = ForensicReportBuilder()
        report = builder.build(
            file_path=tmp_path,
            visual_result=visual_result,
            audio_result=audio_result,
            speaker_result=speaker_result,
            sync_result=sync_result,
            provenance_result=provenance_result,
        )
        
        # Generate PDF
        pdf_path = builder.generate_pdf(report)
        
        print(f"[OK] Forensic report generated successfully!")
        print(f"  Case ID: {report.case_info.case_id}")
        print(f"  Verdict: {report.overall_verdict}")
        print(f"  Trust Score: {report.trust_score:.1f}/100")
        print(f"  Confidence: {report.overall_confidence:.1%}")
        print(f"  PDF saved to: {pdf_path}")
        
        # Verify PDF exists
        assert os.path.exists(pdf_path), "PDF file not created"
        
        # Check PDF size
        pdf_size = os.path.getsize(pdf_path)
        print(f"  PDF size: {pdf_size / 1024:.2f} KB")
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("Testing forensic report generation...")
    print("=" * 50)
    
    try:
        success = test_forensic_report_generation()
        print("=" * 50)
        if success:
            print("[OK] All tests passed!")
        else:
            print("[FAIL] Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
