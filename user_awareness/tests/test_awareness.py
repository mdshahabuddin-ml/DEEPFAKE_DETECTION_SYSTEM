"""Tests for user awareness module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from user_awareness.educational_content import EducationalContent
from user_awareness.limitations_doc import LimitationsDocument
from user_awareness.best_practices import BestPractices


def test_educational_topics():
    print("Testing educational topics...")
    topics = EducationalContent.get_all_topics()
    assert len(topics) == 6
    assert "what_is_deepfake" in topics
    print("  [PASS] All educational topics available")


def test_educational_format():
    print("Testing educational format...")
    text = EducationalContent.get_topic("what_is_deepfake")
    assert isinstance(text, str)
    assert len(text) > 100
    print("  [PASS] Topic content valid")


def test_educational_display():
    print("Testing educational display format...")
    text = "Line 1\n- Bullet point\nLine 3"
    formatted = EducationalContent.format_for_display(text)
    assert "  -" in formatted
    print("  [PASS] Display formatting works")


def test_limitations_sections():
    print("Testing limitations sections...")
    sections = LimitationsDocument.get_all_sections()
    assert len(sections) == 7
    assert "general" in sections
    print("  [PASS] All limitation sections available")


def test_limitations_content():
    print("Testing limitations content...")
    content = LimitationsDocument.get_section("false_positives")
    assert "False Positive" in content
    print("  [PASS] Content is relevant")


def test_limitations_summary():
    print("Testing limitations summary...")
    summary = LimitationsDocument.get_summary()
    assert "SUMMARY" in summary
    print("  [PASS] Summary generated")


def test_best_practices_categories():
    print("Testing best practices categories...")
    categories = BestPractices.get_all_categories()
    assert len(categories) == 5
    assert "verification" in categories
    print("  [PASS] All categories available")


def test_best_practices_checklist():
    print("Testing best practices checklist...")
    checklist = BestPractices.get_category("verification")
    assert len(checklist) > 0
    assert isinstance(checklist, list)
    print("  [PASS] Checklist has items")


def test_best_practices_quick_tips():
    print("Testing best practices quick tips...")
    tips = BestPractices.get_quick_tips()
    assert len(tips) == 6
    print("  [PASS] Quick tips available")


def test_best_practices_format():
    print("Testing best practices format...")
    items = ["Item 1", "Item 2", "Item 3"]
    formatted = BestPractices.format_checklist(items)
    assert "1. Item 1" in formatted
    print("  [PASS] Formatting works")


def run_all_tests():
    print("=" * 60)
    print("USER AWARENESS TESTS")
    print("=" * 60)

    tests = [
        test_educational_topics,
        test_educational_format,
        test_educational_display,
        test_limitations_sections,
        test_limitations_content,
        test_limitations_summary,
        test_best_practices_categories,
        test_best_practices_checklist,
        test_best_practices_quick_tips,
        test_best_practices_format,
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
