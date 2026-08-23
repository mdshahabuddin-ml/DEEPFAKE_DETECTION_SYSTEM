import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setError("");
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please select an image or video first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const isVideo = file.type.startsWith("video/");

      const endpoint = isVideo
        ? `${API_URL}/predict/video`
        : `${API_URL}/predict/image`;

      const formData = new FormData();

      // IMPORTANT:
      // FastAPI expects the field name "file"
      formData.append("file", file);

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || `Backend returned HTTP ${response.status}`
        );
      }

      setResult(data);
    } catch (err) {
      console.error("Analysis error:", err);

      setError(
        err.message ||
          "Unable to connect to the backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      {/* HEADER */}
      <header className="header">
        <div>
          <h1>Deepfake Detection System</h1>
          <p>AI-Powered Explainable Deepfake Detection</p>
        </div>

        <div className="backend-status">
          <span className="status-dot"></span>
          Backend Online
        </div>
      </header>

      {/* MAIN */}
      <main className="main">

        {/* HERO */}
        <section className="hero">
          <div className="badge">
            AI / Xception
          </div>

          <h2>
            Detect Deepfakes
            <span> with AI</span>
          </h2>

          <p>
            Upload an image or video and let the trained
            deepfake detection model analyze it.
          </p>
        </section>

        {/* UPLOAD */}
        <section className="upload-card">

          <div className="upload-icon">
            ↑
          </div>

          <h3>Upload Media</h3>

          <p className="upload-description">
            Select an image or video for deepfake analysis.
          </p>

          <label className="choose-button">
            Choose Image / Video

            <input
              type="file"
              accept="image/jpeg,image/png,image/jpg,video/mp4,video/avi,video/mov,video/mkv"
              onChange={handleFileChange}
            />
          </label>

          {file && (
            <div className="file-info">
              <div>
                <strong>Selected File</strong>
                <span>{file.name}</span>
              </div>

              <div>
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </div>
            </div>
          )}

          <button
            className="analyze-button"
            disabled={!file || loading}
            onClick={handleAnalyze}
          >
            {loading
              ? "Analyzing..."
              : file
                ? "Analyze Media"
                : "Select a File First"}
          </button>

          {/* ERROR */}
          {error && (
            <div className="error-box">
              <strong>Analysis Error</strong>
              <span>{error}</span>
            </div>
          )}

        </section>

        {/* RESULT */}
        {result && (
          <section className="result-card">

            <div className="result-header">
              <h2>Analysis Result</h2>

              <div
                className={`prediction-badge ${
                  result.prediction === "FAKE"
                    ? "fake"
                    : result.prediction === "REAL"
                      ? "real"
                      : "unknown"
                }`}
              >
                {result.prediction}
              </div>
            </div>

            <div className="result-grid">

              <div className="result-item">
                <span>Prediction</span>
                <strong>{result.prediction ?? "N/A"}</strong>
              </div>

              <div className="result-item">
                <span>Fake Probability</span>
                <strong>
                  {result.fake_probability != null
                    ? `${(result.fake_probability * 100).toFixed(2)}%`
                    : "N/A"}
                </strong>
              </div>

              <div className="result-item">
                <span>Confidence</span>
                <strong>
                  {result.confidence != null
                    ? `${(result.confidence * 100).toFixed(2)}%`
                    : "N/A"}
                </strong>
              </div>

              <div className="result-item">
                <span>Media Type</span>
                <strong>{result.media_type ?? "N/A"}</strong>
              </div>

              {result.frames_analyzed != null && (
                <div className="result-item">
                  <span>Frames Analyzed</span>
                  <strong>{result.frames_analyzed}</strong>
                </div>
              )}

              {result.inference_time != null && (
                <div className="result-item">
                  <span>Inference Time</span>
                  <strong>
                    {Number(result.inference_time).toFixed(2)} sec
                  </strong>
                </div>
              )}

            </div>

          </section>
        )}

        {/* FEATURES */}
        <section className="features">

          <div className="feature-card">
            <div className="feature-icon">🧠</div>

            <h3>AI Detection</h3>

            <p>
              Xception-based deepfake classification
              using your trained model.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">📊</div>

            <h3>Probability</h3>

            <p>
              View fake probability, confidence and
              prediction results.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🔍</div>

            <h3>Explainable AI</h3>

            <p>
              Understand the evidence behind the
              deepfake prediction.
            </p>
          </div>

        </section>

        {/* SYSTEM INFORMATION */}
        <section className="system-card">

          <h3>System Status</h3>

          <div className="system-grid">

            <div>
              <span>Frontend</span>
              <strong>Online</strong>
            </div>

            <div>
              <span>Backend</span>
              <strong>Online</strong>
            </div>

            <div>
              <span>Model</span>
              <strong>Xception</strong>
            </div>

            <div>
              <span>Threshold</span>
              <strong>0.47</strong>
            </div>

          </div>

        </section>

      </main>

      <footer>
        Deepfake Detection System • AI-Powered Analysis
      </footer>

    </div>
  );
}

export default App;
