def run_image(detector, uploaded_file):
    """
    Run image prediction and normalize the detector result
    for the Streamlit UI.
    """

    import tempfile
    import time
    from pathlib import Path

    suffix = Path(uploaded_file.name).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        image_path = tmp.name

    start = time.perf_counter()

    try:
        raw = detector.predict_image(image_path)

        inference_time = time.perf_counter() - start

        # Detector uses probability_fake.
        # UI historically used fake_probability.
        probability_fake = float(
            raw.get(
                "probability_fake",
                raw.get("fake_probability", 0.0),
            )
        )

        prediction = raw.get("prediction")

        if prediction is None:
            prediction = "FAKE" if (
                probability_fake >= detector.threshold
            ) else "REAL"

        result = {
            **raw,

            # Canonical UI fields
            "prediction": prediction,
            "probability_fake": probability_fake,
            "fake_probability": probability_fake,
            "confidence": float(
                raw.get("confidence", probability_fake)
            ),
            "inference_time": float(
                raw.get("inference_time", inference_time)
            ),
        }

        return result

    finally:
        try:
            Path(image_path).unlink(missing_ok=True)
        except Exception:
            pass
