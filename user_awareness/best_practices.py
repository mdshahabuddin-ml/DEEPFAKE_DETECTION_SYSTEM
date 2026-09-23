"""Best practices for media verification."""


class BestPractices:
    """Best practices for media verification."""

    VERIFICATION_CHECKLIST = [
        "Check the source before sharing",
        "Look for multiple independent confirmations",
        "Be skeptical of emotionally charged content",
        "Verify timestamps and context",
        "Consult fact-checking organizations",
        "Consider who benefits from the content",
    ]

    SHARING_GUIDELINES = [
        "Don't share unverified content",
        "Add context when sharing",
        "Consider impact before sharing",
        "Be transparent about uncertainties",
        "Correct mistakes quickly",
        "Protect privacy of others",
    ]

    CRITICAL_THINKING = [
        "Who benefits from this content?",
        "What is the evidence?",
        "Are there alternative explanations?",
        "What is missing from this story?",
        "How does this make me feel?",
        "Can this be independently verified?",
    ]

    REPORTING_STEPS = [
        "Document evidence (save original if possible)",
        "Report to the platform using official tools",
        "Alert fact-checking organizations",
        "Avoid amplifying the fake content",
        "Seek expert help if needed",
        "Notify affected parties if appropriate",
    ]

    PROTECTION_TIPS = [
        "Limit public photos and videos",
        "Review privacy settings regularly",
        "Use strong, unique passwords",
        "Enable two-factor authentication",
        "Monitor for unauthorized use of your likeness",
        "Know your legal rights",
    ]

    @classmethod
    def get_all_categories(cls) -> dict:
        return {
            "verification": cls.VERIFICATION_CHECKLIST,
            "sharing": cls.SHARING_GUIDELINES,
            "critical_thinking": cls.CRITICAL_THINKING,
            "reporting": cls.REPORTING_STEPS,
            "protection": cls.PROTECTION_TIPS,
        }

    @classmethod
    def get_category(cls, name: str) -> list:
        categories = cls.get_all_categories()
        return categories.get(name, [])

    @classmethod
    def get_quick_tips(cls) -> list:
        return [
            "Check the source before sharing",
            "Look for multiple confirmations",
            "Be skeptical of emotional content",
            "Verify timestamps and context",
            "Consult fact-checking sites",
            "Consider who benefits",
        ]

    @classmethod
    def format_checklist(cls, items: list) -> str:
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines)
