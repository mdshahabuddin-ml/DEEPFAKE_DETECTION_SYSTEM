"""Documentation about detector limitations."""


class LimitationsDocument:
    """Document explaining detector limitations."""

    GENERAL_LIMITATIONS = """
General Limitations of Deepfake Detectors

No Detection System is Perfect:

1. Accuracy is Not 100%
   - False Positives: Authentic media flagged as fake
   - False Negatives: Fake media not detected
   - Both types of errors have consequences

2. Performance Varies by Content Type
   - Face-focused videos: Generally better detection
   - Full-body or distant shots: Lower accuracy
   - Audio-only: Different challenges than video
   - Low quality: May reduce detection performance

3. Temporal Limitations
   - Detectors are trained on current manipulation techniques
   - New methods may evade detection
   - Regular model updates are needed

4. Environmental Factors
   - Lighting conditions affect performance
   - Camera angles and quality matter
   - Compression and format changes impact results
"""

    FALSE_POSITIVES = """
Understanding False Positives

What is a False Positive?
- Authentic content incorrectly flagged as manipulated
- Can cause unwarranted suspicion
- May damage reputations

Why They Happen:
- Unusual lighting or camera angles
- Heavy makeup or facial hair changes
- Video compression artifacts
- Low-quality recordings
- Natural facial variations

Impact:
- Loss of trust in detection systems
- Potential harm to innocent people
- Resource waste on investigations

Mitigation:
- Use multiple detection methods
- Set appropriate confidence thresholds
- Human review for borderline cases
"""

    FALSE_NEGATIVES = """
Understanding False Negatives

What is a False Negative?
- Manipulated content not detected
- Fake media passes as authentic
- Undermines detection system reliability

Why They Happen:
- High-quality manipulations
- Novel attack methods
- Adversarial perturbations
- Domain shift from training data

Impact:
- Spread of misinformation
- Potential harm from fake content
- Erosion of public trust

Mitigation:
- Continuous model updates
- Ensemble methods
- Human expert review
- Multi-modal analysis
"""

    DATASET_BIAS = """
Understanding Dataset Bias

What is Dataset Bias?
- Training data doesn't represent all scenarios
- Certain demographics may be over/under-represented
- Specific manipulation types dominate training

Common Biases:
- Face-centric datasets
- Western-centric faces
- Specific manipulation methods
- High-quality examples preferred

Impact:
- Performance varies across demographics
- Some manipulation types harder to detect
- Generalization to real-world data affected

Mitigation:
- Diverse training datasets
- Regular bias audits
- Performance monitoring across groups
- Transparent reporting of limitations
"""

    DOMAIN_SHIFT = """
Understanding Domain Shift

What is Domain Shift?
- Difference between training and deployment data
- Real-world media varies from controlled datasets

Examples:
- Compression differences
- Resolution variations
- Recording device differences
- Environmental conditions

Impact:
- Reduced detection accuracy
- Unpredictable performance
- Need for domain adaptation

Mitigation:
- Data augmentation during training
- Domain adaptation techniques
- Regular performance monitoring
- Continuous model retraining
"""

    COMPRESSION_EFFECTS = """
Understanding Compression Effects

How Compression Affects Detection:
- Removes high-frequency details
- Introduces block artifacts
- Reduces overall quality
- Can mask manipulation artifacts

Impact on Detection:
- Both fake and real media are affected
- May hide manipulation traces
- Reduces feature discriminability
- Performance degrades with heavy compression

Considerations:
- Detection confidence may decrease
- Multiple quality levels should be tested
- Original quality when available is preferred
"""

    RECORDING_QUALITY = """
Understanding Recording Quality Effects

Quality Variations:
- Resolution (4K vs 480p)
- Frame rate (60fps vs 24fps)
- Lighting conditions
- Audio quality
- Camera stability

Impact on Detection:
- Low quality reduces available features
- Noise can mask manipulation artifacts
- High quality may reveal more details
- Optimal conditions improve accuracy

Recommendations:
- Use highest quality available
- Test multiple quality levels
- Consider quality in confidence assessment
"""

    @classmethod
    def get_all_sections(cls) -> dict:
        """Get all limitation sections."""
        return {
            "general": cls.GENERAL_LIMITATIONS,
            "false_positives": cls.FALSE_POSITIVES,
            "false_negatives": cls.FALSE_NEGATIVES,
            "dataset_bias": cls.DATASET_BIAS,
            "domain_shift": cls.DOMAIN_SHIFT,
            "compression": cls.COMPRESSION_EFFECTS,
            "recording_quality": cls.RECORDING_QUALITY,
        }

    @classmethod
    def get_section(cls, section_name: str) -> str:
        """Get specific section content."""
        sections = cls.get_all_sections()
        return sections.get(section_name, "Section not found")

    @classmethod
    def get_summary(cls) -> str:
        """Get summary of all limitations."""
        sections = cls.get_all_sections()
        summary = "DETECTOR LIMITATIONS SUMMARY\n" + "=" * 50 + "\n\n"
        for name, content in sections.items():
            summary += f"## {name.upper()}\n"
            summary += content[:200] + "...\n\n"
        return summary
