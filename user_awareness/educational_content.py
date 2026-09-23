"""Educational content about deepfakes and media manipulation."""


class EducationalContent:
    """Educational content about deepfakes and media manipulation."""

    WHAT_IS_DEEPFAKE = """
What is a Deepfake?

A deepfake is synthetic media where a person's likeness is replaced
with someone else's using artificial intelligence (AI) and machine
learning techniques.

Key Points:
- Deepfakes use deep learning neural networks
- They can swap faces, change voices, or generate entirely new content
- Quality varies from obvious to very convincing
- Both images and videos can be manipulated
- Audio deepfakes (voice cloning) are also possible
"""

    HOW_AUDIO_MANIPULATED = """
How Can Audio Be Manipulated?

Audio manipulation techniques include:

1. Voice Cloning
   - AI models can learn a person's voice from samples
   - New speech can be generated in their voice
   - Quality depends on amount of training data

2. Text-to-Speech (TTS)
   - Convert written text to spoken audio
   - Modern TTS can sound very natural
   - Can be used to create fake statements

3. Voice Conversion
   - Change one person's voice to sound like another
   - Preserves speech patterns and emotion
   - Used in both legitimate and malicious applications

4. Audio Splicing
   - Cut and rearrange audio segments
   - Create fake statements from real recordings
   - Traditional technique still effective
"""

    WHAT_IS_LIP_SYNC = """
What is Lip-Sync Manipulation?

Lip-sync manipulation makes a person appear to say things they never said.

How it Works:
- AI analyzes mouth movements in video
- Generates new mouth movements to match fake audio
- Creates illusion of person speaking fabricated words

Detection Challenges:
- High-quality manipulations are hard to detect
- Inconsistencies may appear around mouth area
- Lighting and angle can hide artifacts
- Requires careful visual inspection
"""

    WHY_DETECTORS_MISTAKE = """
Why Can Detectors Make Mistakes?

No detection system is perfect. Here's why:

1. Training Data Limitations
   - Models are trained on specific types of deepfakes
   - New manipulation techniques may not be represented
   - Dataset bias can affect performance

2. Domain Shift
   - Real-world media differs from training data
   - Compression, quality, and format variations
   - Different recording conditions

3. Adversarial Attacks
   - Manipulators can intentionally fool detectors
   - Small perturbations can change predictions
   - Arms race between creators and detectors

4. Quality Variations
   - Low-quality deepfakes are easier to detect
   - High-quality ones may pass detection
   - Natural variation in manipulation quality

5. Edge Cases
   - Unusual lighting, angles, or expressions
   - Multiple people in frame
    Partial face visibility
"""

    VERIFY_INDEPENDENTLY = """
Why Verify Important Claims Independently?

Critical Reasoning:
- Single sources can be biased or manipulated
- Multiple independent sources increase confidence
- Official channels should be consulted
- Context matters for interpretation

Verification Steps:
1. Check multiple news sources
2. Look for official statements
3. Verify the original source
4. Check timestamps and context
5. Consult fact-checking organizations
6. Consider who benefits from the content

Remember:
- Viral content is not necessarily true
- Emotional reactions can cloud judgment
- Take time to verify before sharing
"""

    WHY_NOT_RELY_SINGLE = """
Why Not Rely on a Single AI Detector?

Multiple Perspectives:
- Different detectors use different approaches
- No single method catches all manipulations
- Combining detectors improves reliability
- Cross-validation increases confidence

Technical Reasons:
- Each detector has blind spots
- Training data varies between models
- Different features are analyzed
- Ensemble methods perform better

Practical Advice:
- Use multiple detection tools
- Consider detector confidence levels
- Look for corroborating evidence
- Apply human judgment alongside AI
"""

    @classmethod
    def get_all_topics(cls) -> dict:
        """Get all educational topics."""
        return {
            "what_is_deepfake": cls.WHAT_IS_DEEPFAKE,
            "audio_manipulation": cls.HOW_AUDIO_MANIPULATED,
            "lip_sync": cls.WHAT_IS_LIP_SYNC,
            "detector_mistakes": cls.WHY_DETECTORS_MISTAKE,
            "verify_independently": cls.VERIFY_INDEPENDENTLY,
            "multiple_detectors": cls.WHY_NOT_RELY_SINGLE,
        }

    @classmethod
    def get_topic(cls, topic_name: str) -> str:
        """Get specific topic content."""
        topics = cls.get_all_topics()
        return topics.get(topic_name, "Topic not found")

    @classmethod
    def format_for_display(cls, text: str) -> str:
        """Format content for display."""
        lines = text.strip().split("\n")
        formatted = []
        for line in lines:
            if line.startswith("- "):
                formatted.append(f"  {line}")
            elif line.strip():
                formatted.append(line)
        return "\n".join(formatted)
