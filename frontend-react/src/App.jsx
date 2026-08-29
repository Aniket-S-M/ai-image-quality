import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Database,
  FileImage,
  Info,
  Layers3,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Upload,
  XCircle,
  Zap,
} from "lucide-react";

import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const STAT_LABELS = {
  sharpness: "Sharpness",
  brightness: "Brightness",
  contrast: "Contrast",
  saturation: "Saturation",
  edge_density: "Edge Density",
  noise_estimate: "Noise Estimate",
  highlight_clipping: "Highlight Clipping",
  shadow_clipping: "Shadow Clipping",
  dark_pixel_ratio: "Dark Pixel Ratio",
  bright_pixel_ratio: "Bright Pixel Ratio",
  blockiness: "Blockiness",
};

/*
 * ============================================================
 * PRODUCTION MODEL
 * ============================================================
 *
 * These values come from the final expanded RF evaluation.
 *
 * Dataset:
 *   Total      : 9,500
 *   Train      : 6,650
 *   Validation : 1,425
 *   Test       : 1,425
 *
 * Test:
 *   Accuracy   : 88.21%
 *   Macro F1   : 83.64%
 */

const RF_METRICS = {
  accuracy: 88.21,
  f1: 83.64,
};

const DATASET_STATS = {
  total: 9500,
  train: 6650,
  validation: 1425,
  test: 1425,
  features: 10,
};

const CLASS_RESULTS = [
  { name: "Blur", f1: 0.95 },
  { name: "Corruption", f1: 0.78 },
  { name: "Noise", f1: 0.91 },
  { name: "None", f1: 0.54 },
  { name: "Overexposure", f1: 0.85 },
  { name: "Underexposure", f1: 0.98 },
];

const CLASS_NAMES = [
  "Blur",
  "Corruption",
  "Noise",
  "None",
  "Overexposure",
  "Underexposure",
];

/*
 * Final test confusion matrix.
 *
 * Rows    = actual class
 * Columns = predicted class
 */
const CONFUSION_MATRIX = [
  [217, 3, 0, 0, 0, 5],
  [11, 167, 24, 12, 10, 1],
  [0, 14, 426, 2, 8, 0],
  [2, 8, 18, 38, 8, 1],
  [0, 11, 16, 13, 185, 0],
  [1, 0, 0, 0, 0, 224],
];

/*
 * Global Random Forest feature importance.
 *
 * These describe the model globally.
 * They do NOT claim that each feature caused
 * an individual prediction.
 */
const FEATURE_IMPORTANCE = [
  ["Sharpness", 24.14],
  ["Highlight Clipping", 16.42],
  ["Bright Pixel Ratio", 15.57],
  ["Blockiness", 13.53],
  ["Edge Density", 9.33],
  ["Brightness", 6.07],
  ["Contrast", 5.2],
  ["Shadow Clipping", 3.49],
  ["Dark Pixel Ratio", 3.16],
  ["Saturation", 3.1],
];

function App() {
  const [apiOnline, setApiOnline] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const [selectedHistory, setSelectedHistory] = useState(null);
  const [historyDetailLoading, setHistoryDetailLoading] =
    useState(false);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  useEffect(() => {
    checkAPI();
    loadHistory();

    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  async function checkAPI() {
    try {
      const response = await fetch(`${API_URL}/health`);

      if (!response.ok) {
        throw new Error("API unavailable");
      }

      const healthData = await response.json();

      setApiOnline(true);

      /*
       * Health endpoint confirms connectivity.
       * Model metadata comes from /model.
       */
      try {
        const modelResponse = await fetch(
          `${API_URL}/model`
        );

        if (modelResponse.ok) {
          const modelData = await modelResponse.json();
          setModelInfo(modelData);
        } else {
          setModelInfo(healthData);
        }
      } catch {
        setModelInfo(healthData);
      }
    } catch {
      setApiOnline(false);
      setModelInfo(null);
    }
  }

  async function loadHistory() {
    setHistoryLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/analyses`
      );

      if (!response.ok) {
        throw new Error("Could not load history");
      }

      const data = await response.json();

      setHistory(data.analyses || []);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function openHistoryRecord(id) {
    setHistoryDetailLoading(true);
    setSelectedHistory(null);

    try {
      const response = await fetch(
        `${API_URL}/analyses/${id}`
      );

      if (!response.ok) {
        throw new Error(
          "Could not load analysis details."
        );
      }

      const data = await response.json();

      setSelectedHistory(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not load analysis details."
      );
    } finally {
      setHistoryDetailLoading(false);
    }
  }

  function handleFile(selectedFile) {
    setError("");

    if (!selectedFile) {
      return;
    }

    const allowed = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    if (!allowed.includes(selectedFile.type)) {
      setError(
        "Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP."
      );
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(selectedFile);
    setPreview(
      URL.createObjectURL(selectedFile)
    );
    setResult(null);
  }

  function handleInputChange(event) {
    handleFile(event.target.files?.[0]);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);

    handleFile(
      event.dataTransfer.files?.[0]
    );
  }

  async function analyzeImage() {
    if (!file) {
      setError(
        "Please choose an image first."
      );
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/analyze`,
        {
          method: "POST",
          body: formData,
        }
      );

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The server returned an invalid response."
        );
      }

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Image analysis failed."
        );
      }

      setResult(data);

      await loadHistory();
    } catch (err) {
      setError(
        err.message ||
          "Could not analyze the image."
      );
    } finally {
      setLoading(false);
    }
  }

  function clearImage() {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  const isAccepted =
    result?.decision === "ACCEPTED";

  return (
    <div className="app-shell">
      {/* ======================================================
          TOP BAR
          ====================================================== */}

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <ScanSearch size={22} />
          </div>

          <div>
            <div className="brand-name">
              VisionIQ
            </div>

            <div className="brand-subtitle">
              AI Image Quality Intelligence
            </div>
          </div>
        </div>

        <div className="topbar-actions">
          <div
            className={`api-pill ${
              apiOnline
                ? "online"
                : "offline"
            }`}
          >
            <span className="status-dot" />

            {apiOnline
              ? "API Connected"
              : "API Offline"}
          </div>

          <button
            className="icon-button"
            onClick={() => {
              checkAPI();
              loadHistory();
            }}
            title="Refresh system"
          >
            <RefreshCw size={17} />
          </button>
        </div>
      </header>

      <main>
        {/* ====================================================
            HERO
            ==================================================== */}

        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <Sparkles size={14} />
              COMPUTER VISION + MACHINE LEARNING
            </div>

            <h1>
              Turn visual quality
              <br />
              into{" "}
              <span>
                measurable intelligence.
              </span>
            </h1>

            <p>
              Analyze image degradation using
              interpretable computer-vision
              features, Random Forest V3
              Expanded, and Isolation Forest
              anomaly detection.
            </p>

            <div className="hero-stats">
              <div>
                <strong>
                  {RF_METRICS.accuracy.toFixed(
                    2
                  )}
                  %
                </strong>

                <span>
                  RF V3 Expanded accuracy
                </span>
              </div>

              <div>
                <strong>
                  {RF_METRICS.f1.toFixed(2)}%
                </strong>

                <span>
                  Test Macro-F1
                </span>
              </div>

              <div>
                <strong>
                  {DATASET_STATS.total.toLocaleString()}
                </strong>

                <span>
                  Dataset images
                </span>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="orb orb-one" />
            <div className="orb orb-two" />

            <div className="architecture-card">
              <div className="architecture-header">
                <span>
                  QUALITY PIPELINE
                </span>

                <Activity size={17} />
              </div>

              <PipelineStep
                number="01"
                icon={
                  <FileImage size={17} />
                }
                title="Image Upload"
                text="JPEG · PNG · WEBP"
              />

              <PipelineStep
                number="02"
                icon={
                  <Layers3 size={17} />
                }
                title="CV Features"
                text="10 measurable signals"
              />

              <PipelineStep
                number="03"
                icon={
                  <Brain size={17} />
                }
                title="RF V3 Expanded"
                text="Degradation classification"
              />

              <PipelineStep
                number="04"
                icon={
                  <ShieldCheck size={17} />
                }
                title="Decision Engine"
                text="Score + anomaly detection"
                last
              />
            </div>
          </div>
        </section>

        {/* ====================================================
            ANALYSIS
            ==================================================== */}

        <section
          className="workspace section"
          id="analyze"
        >
          <SectionHeading
            eyebrow="ANALYZE"
            title="Assess an image"
            text="Upload a visual and run the complete quality pipeline."
          />

          <div className="analysis-grid">
            <div className="upload-panel panel">
              {!preview ? (
                <div
                  className={`dropzone ${
                    dragActive
                      ? "drag-active"
                      : ""
                  }`}
                  onDragEnter={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    setDragActive(false);
                  }}
                  onDrop={handleDrop}
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                    onChange={handleInputChange}
                    hidden
                  />

                  <div className="upload-icon">
                    <Upload size={26} />
                  </div>

                  <h3>
                    Drop your image here
                  </h3>

                  <p>
                    or click to browse from
                    your device
                  </p>

                  <div className="format-row">
                    <span>JPG</span>
                    <span>JPEG</span>
                    <span>PNG</span>
                    <span>WEBP</span>
                  </div>
                </div>
              ) : (
                <div className="preview-area">
                  <div className="preview-toolbar">
                    <div>
                      <strong>
                        Image Preview
                      </strong>

                      <span>
                        {file.name}
                      </span>
                    </div>

                    <button
                      className="remove-button"
                      onClick={clearImage}
                    >
                      <XCircle size={18} />
                    </button>
                  </div>

                  <div className="image-frame">
                    <img
                      src={preview}
                      alt="Uploaded preview"
                    />
                  </div>

                  <div className="file-meta">
                    <span>
                      <FileImage size={15} />
                      {formatBytes(file.size)}
                    </span>

                    <span>
                      {file.type
                        .replace(
                          "image/",
                          ""
                        )
                        .toUpperCase()}
                    </span>
                  </div>
                </div>
              )}

              {error && (
                <div className="error-box">
                  <AlertTriangle size={17} />
                  <span>{error}</span>
                </div>
              )}

              <button
                className="analyze-button"
                disabled={!file || loading}
                onClick={analyzeImage}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analyzing image...
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    Run Quality Analysis
                  </>
                )}
              </button>
            </div>

            <div className="result-panel">
              {!result ? (
                <EmptyResult />
              ) : (
                <AssessmentResult
                  result={result}
                  isAccepted={isAccepted}
                />
              )}
            </div>
          </div>
        </section>

        {/* ====================================================
            LIVE RESULT DETAILS
            ==================================================== */}

        {result && (
          <>
            <section className="section">
              <SectionHeading
                eyebrow="DIAGNOSTICS"
                title="Visual diagnostics"
                text="Measured image statistics extracted before classification."
              />

              <StatisticsPanel
                statistics={
                  result.image_statistics
                }
              />
            </section>

            <section className="section">
              <AnomalyPanel
                anomaly={result.anomaly}
              />
            </section>

            <section className="section">
              <ExplainabilityPanel
                result={result}
              />
            </section>
          </>
        )}

        {/* ====================================================
            EVALUATION
            ==================================================== */}

        <section
          className="section"
          id="evaluation"
        >
          <SectionHeading
            eyebrow="EVALUATION"
            title="Model performance"
            text="Final held-out evaluation of the expanded Random Forest classifier."
          />

          <EvaluationSection />
        </section>

        {/* ====================================================
            RELIABILITY
            ==================================================== */}

        <section className="section">
          <SectionHeading
            eyebrow="RELIABILITY"
            title="Failure cases & limitations"
            text="Transparent reporting of where the system can be uncertain."
          />

          <LimitationsSection />
        </section>

        {/* ====================================================
            HISTORY
            ==================================================== */}

        <section
          className="section"
          id="history"
        >
          <SectionHeading
            eyebrow="HISTORY"
            title="Previous analyses"
            text="Results persisted through the FastAPI + SQLite backend. Click an analysis to inspect its stored result."
          />

          <HistorySection
            history={history}
            loading={historyLoading}
            onOpen={openHistoryRecord}
          />
        </section>

        {/* ====================================================
            ABOUT
            ==================================================== */}

        <section className="section">
          <AboutSection
            modelInfo={modelInfo}
          />
        </section>
      </main>

      {/* ======================================================
          HISTORY DETAIL MODAL
          ====================================================== */}

      {(selectedHistory ||
        historyDetailLoading) && (
        <HistoryModal
          record={selectedHistory}
          loading={historyDetailLoading}
          onClose={() =>
            setSelectedHistory(null)
          }
        />
      )}

      {/* ======================================================
          FOOTER
          ====================================================== */}

      <footer>
        <div>
          <strong>VisionIQ</strong>

          <span>
            AI Image Quality Analyzer
          </span>
        </div>

        <div>
          Random Forest V3 Expanded ·
          Computer Vision · Isolation Forest
        </div>
      </footer>
    </div>
  );
}

/* ============================================================
   PIPELINE
   ============================================================ */

function PipelineStep({
  number,
  icon,
  title,
  text,
  last = false,
}) {
  return (
    <div
      className={`pipeline-step ${
        last ? "last" : ""
      }`}
    >
      <div className="pipeline-number">
        {number}
      </div>

      <div className="pipeline-icon">
        {icon}
      </div>

      <div>
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  );
}

/* ============================================================
   SECTION HEADING
   ============================================================ */

function SectionHeading({
  eyebrow,
  title,
  text,
}) {
  return (
    <div className="section-heading">
      <div className="eyebrow">
        {eyebrow}
      </div>

      <h2>{title}</h2>

      <p>{text}</p>
    </div>
  );
}

/* ============================================================
   EMPTY RESULT
   ============================================================ */

function EmptyResult() {
  return (
    <div className="empty-result">
      <div className="empty-icon">
        <ScanSearch size={30} />
      </div>

      <h3>
        Ready for analysis
      </h3>

      <p>
        Upload an image and run the
        quality pipeline. Your score,
        decision, detected issues,
        diagnostics and anomaly signal
        will appear here.
      </p>

      <div className="empty-flow">
        <span>Features</span>

        <ChevronRight size={15} />

        <span>Classification</span>

        <ChevronRight size={15} />

        <span>Decision</span>
      </div>
    </div>
  );
}

/* ============================================================
   ASSESSMENT RESULT
   ============================================================ */

function AssessmentResult({
  result,
  isAccepted,
}) {
  return (
    <div className="assessment-card">
      <div className="assessment-top">
        <div>
          <div className="eyebrow">
            QUALITY ASSESSMENT
          </div>

          <h2>
            {isAccepted
              ? "Image accepted"
              : "Image requires attention"}
          </h2>

          <p>
            {isAccepted
              ? "This image meets the configured quality acceptance threshold."
              : "The image falls below the configured quality acceptance threshold."}
          </p>
        </div>

        <div
          className={`decision-badge ${
            isAccepted
              ? "accepted"
              : "rejected"
          }`}
        >
          {isAccepted ? (
            <CheckCircle2 size={18} />
          ) : (
            <XCircle size={18} />
          )}

          {result.decision}
        </div>
      </div>

      <div className="score-display">
        <div className="score-number">
          {result.quality_score}
        </div>

        <div className="score-denom">
          /100
        </div>

        <div className="score-label">
          {result.quality_label}
        </div>
      </div>

      <div className="score-track">
        <div
          className={`score-fill ${
            result.quality_score >= 75
              ? "good"
              : result.quality_score >= 50
              ? "medium"
              : "bad"
          }`}
          style={{
            width: `${Math.max(
              0,
              Math.min(
                100,
                result.quality_score
              )
            )}%`,
          }}
        />
      </div>

      <div className="threshold-note">
        <ShieldCheck size={15} />
        Acceptance threshold:
        <strong>75/100</strong>
      </div>

      <div className="issue-summary">
        <div className="mini-heading">
          DETECTED ISSUES
        </div>

        {result.issues?.length ? (
          <div className="issue-list">
            {result.issues.map(
              (issue, index) => (
                <IssueCard
                  issue={issue}
                  key={`${issue.type}-${index}`}
                />
              )
            )}
          </div>
        ) : (
          <div className="no-issues">
            <CheckCircle2 size={18} />
            No known quality issues
            detected.
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================
   ISSUE CARD
   ============================================================ */

function IssueCard({ issue }) {
  const severity =
    issue.severity || "unknown";

  return (
    <div className="issue-card">
      <div className="issue-main">
        <div className="issue-icon">
          <AlertTriangle size={17} />
        </div>

        <div>
          <strong>
            {capitalize(issue.type)}
          </strong>

          <span>
            Detected degradation
          </span>
        </div>
      </div>

      <div className="issue-detail">
        <span>Severity</span>

        <strong
          className={`severity ${severity}`}
        >
          {capitalize(severity)}
        </strong>
      </div>

      <div className="issue-detail">
        <span>Confidence</span>

        <strong>
          {(
            (issue.confidence || 0) *
            100
          ).toFixed(1)}
          %
        </strong>
      </div>
    </div>
  );
}

/* ============================================================
   STATISTICS
   ============================================================ */

function StatisticsPanel({
  statistics,
}) {
  if (
    !statistics ||
    Object.keys(statistics).length === 0
  ) {
    return (
      <div className="panel empty-small">
        Image statistics are not
        available for this analysis.
      </div>
    );
  }

  return (
    <div className="stats-grid">
      {Object.entries(
        statistics
      ).map(([key, value]) => (
        <div
          className="stat-card"
          key={key}
        >
          <div className="stat-icon">
            <Activity size={16} />
          </div>

          <span>
            {STAT_LABELS[key] ||
              capitalize(key)}
          </span>

          <strong>
            {formatStat(value)}
          </strong>
        </div>
      ))}
    </div>
  );
}

/* ============================================================
   ANOMALY
   ============================================================ */

function AnomalyPanel({
  anomaly,
}) {
  const detected = Boolean(
    anomaly?.detected
  );

  const score = Number(
    anomaly?.score ?? 0
  );

  return (
    <div className="anomaly-panel">
      <div
        className={`anomaly-icon ${
          detected
            ? "warning"
            : "normal"
        }`}
      >
        {detected ? (
          <AlertTriangle size={24} />
        ) : (
          <ShieldCheck size={24} />
        )}
      </div>

      <div className="anomaly-content">
        <div className="eyebrow">
          ANOMALY DETECTION
        </div>

        <h3>
          {detected
            ? "Unusual visual pattern detected"
            : "No unusual visual pattern detected"}
        </h3>

        <p>
          Isolation Forest provides an
          additional anomaly signal
          beyond the primary degradation
          classifier. It does not replace
          the Random Forest prediction.
        </p>
      </div>

      <div className="anomaly-score">
        <span>
          Isolation Forest score
        </span>

        <strong>
          {score.toFixed(4)}
        </strong>
      </div>
    </div>
  );
}

/* ============================================================
   EXPLAINABILITY
   ============================================================ */

function ExplainabilityPanel({
  result,
}) {
  const issue = result?.issues?.[0];

  const supportingStats =
    getSupportingStatistics(
      issue?.type,
      result?.image_statistics
    );

  return (
    <div className="explainability-stack">
      <div className="explainability-grid">
        {/* Individual prediction explanation */}

        <div className="panel explain-card">
          <div className="card-title">
            <Brain size={18} />
            Why this result?
          </div>

          <div className="decision-flow">
            <FlowBox
              title="CV Features"
              text="10 measured signals"
            />

            <ChevronRight size={17} />

            <FlowBox
              title="RF V3 Expanded"
              text={
                issue
                  ? capitalize(
                      issue.type
                    )
                  : "None"
              }
            />

            <ChevronRight size={17} />

            <FlowBox
              title="Severity"
              text={
                issue
                  ? capitalize(
                      issue.severity
                    )
                  : "None"
              }
            />

            <ChevronRight size={17} />

            <FlowBox
              title="Decision"
              text={result.decision}
            />
          </div>

          <div className="explain-disclaimer">
            This explanation uses the
            measured statistics of this
            image. It does not treat global
            feature importance as a
            per-image causal explanation.
          </div>
        </div>

        {/* Individual image evidence */}

        <div className="panel explain-card">
          <div className="card-title">
            <Info size={18} />
            Image-specific evidence
          </div>

          {issue ? (
            <>
              <p className="explain-text">
                The classifier identified{" "}
                <strong>
                  {capitalize(
                    issue.type
                  )}
                </strong>{" "}
                with{" "}
                <strong>
                  {(
                    issue.confidence *
                    100
                  ).toFixed(1)}
                  %
                </strong>{" "}
                confidence and estimated
                the severity as{" "}
                <strong>
                  {capitalize(
                    issue.severity
                  )}
                </strong>
                .
              </p>

              <div className="supporting-stat-list">
                {supportingStats.map(
                  (item) => (
                    <div
                      className="supporting-stat"
                      key={item.key}
                    >
                      <span>
                        {item.label}
                      </span>

                      <strong>
                        {formatStat(
                          result
                            .image_statistics?.[
                            item.key
                          ]
                        )}
                      </strong>

                      <small>
                        {item.reason}
                      </small>
                    </div>
                  )
                )}
              </div>
            </>
          ) : (
            <p className="explain-text">
              No degradation issue was
              returned by the classifier.
              The image-specific statistics
              are shown in the diagnostics
              section above.
            </p>
          )}
        </div>
      </div>

      {/* Global model explanation */}

      <div className="panel global-importance-card">
        <div className="card-title">
          <BarChart3 size={18} />
          Global Random Forest feature importance
        </div>

        <p className="explain-text">
          These values describe the relative
          contribution of each feature across
          the trained Random Forest model.
          They are global model statistics,
          not a causal explanation for one
          particular image.
        </p>

        <div className="feature-importance-list">
          {FEATURE_IMPORTANCE.map(
            ([name, value]) => (
              <div
                className="feature-importance-row"
                key={name}
              >
                <div>
                  <span>{name}</span>

                  <strong>
                    {value.toFixed(2)}%
                  </strong>
                </div>

                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${value}%`,
                    }}
                  />
                </div>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   SUPPORTING STATISTICS
   ============================================================ */

function getSupportingStatistics(
  issue,
  statistics
) {
  if (!statistics) {
    return [];
  }

  const definitions = {
    underexposure: [
      {
        key: "brightness",
        label: "Brightness",
        reason: "Low overall luminance can support an underexposure finding.",
      },
      {
        key: "shadow_clipping",
        label: "Shadow Clipping",
        reason: "High shadow clipping indicates loss of dark-region detail.",
      },
      {
        key: "dark_pixel_ratio",
        label: "Dark Pixel Ratio",
        reason: "A high proportion of dark pixels supports the exposure assessment.",
      },
      {
        key: "bright_pixel_ratio",
        label: "Bright Pixel Ratio",
        reason: "Very few bright pixels can reinforce the underexposure pattern.",
      },
    ],

    overexposure: [
      {
        key: "brightness",
        label: "Brightness",
        reason: "High luminance can support an overexposure finding.",
      },
      {
        key: "highlight_clipping",
        label: "Highlight Clipping",
        reason: "Clipped highlights indicate loss of bright-region detail.",
      },
      {
        key: "bright_pixel_ratio",
        label: "Bright Pixel Ratio",
        reason: "A high bright-pixel proportion can support the exposure assessment.",
      },
    ],

    blur: [
      {
        key: "sharpness",
        label: "Sharpness",
        reason: "Low sharpness is consistent with reduced image detail.",
      },
      {
        key: "edge_density",
        label: "Edge Density",
        reason: "Lower edge density can accompany loss of fine structure.",
      },
    ],

    noise: [
      {
        key: "noise_estimate",
        label: "Noise Estimate",
        reason: "The noise estimate measures high-frequency intensity variation.",
      },
      {
        key: "sharpness",
        label: "Sharpness",
        reason: "Noise can interact with measured image detail.",
      },
    ],

    corruption: [
      {
        key: "blockiness",
        label: "Blockiness",
        reason: "Block artifacts can indicate structural/compression degradation.",
      },
      {
        key: "edge_density",
        label: "Edge Density",
        reason: "Structural changes can affect local edge statistics.",
      },
    ],

    none: [
      {
        key: "sharpness",
        label: "Sharpness",
        reason: "Sharpness contributes to the overall image-quality profile.",
      },
      {
        key: "contrast",
        label: "Contrast",
        reason: "Contrast contributes to the tonal quality profile.",
      },
    ],
  };

  return (
    definitions[issue] || []
  ).filter(
    (item) =>
      statistics[item.key] !==
      undefined
  );
}

/* ============================================================
   FLOW BOX
   ============================================================ */

function FlowBox({
  title,
  text,
}) {
  return (
    <div className="flow-box">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

/* ============================================================
   EVALUATION
   ============================================================ */

function EvaluationSection() {
  return (
    <div className="evaluation-stack">
      <div className="evaluation-overview-grid">
        <div className="panel evaluation-hero-card">
          <div className="card-title">
            <Brain size={18} />
            Random Forest V3 Expanded
          </div>

          <div className="evaluation-big-metrics">
            <div>
              <strong>
                {RF_METRICS.accuracy.toFixed(
                  2
                )}
                %
              </strong>

              <span>
                Test Accuracy
              </span>
            </div>

            <div>
              <strong>
                {RF_METRICS.f1.toFixed(2)}%
              </strong>

              <span>
                Test Macro-F1
              </span>
            </div>
          </div>

          <div className="selection-note">
            This is the expanded Random
            Forest model used by the current
            quality-analysis backend.
          </div>
        </div>

        <div className="panel dataset-card">
          <div className="card-title">
            <Database size={18} />
            Dataset
          </div>

          <DatasetMetric
            label="Total"
            value={DATASET_STATS.total}
          />

          <DatasetMetric
            label="Training"
            value={DATASET_STATS.train}
          />

          <DatasetMetric
            label="Validation"
            value={DATASET_STATS.validation}
          />

          <DatasetMetric
            label="Test"
            value={DATASET_STATS.test}
          />

          <DatasetMetric
            label="Features"
            value={DATASET_STATS.features}
          />
        </div>
      </div>

      <div className="evaluation-grid">
        <div className="panel class-panel">
          <div className="card-title">
            <Layers3 size={18} />
            Per-class F1 performance
          </div>

          {CLASS_RESULTS.map(
            (item) => (
              <MetricBar
                key={item.name}
                label={item.name}
                value={item.f1 * 100}
              />
            )
          )}
        </div>

        <div className="panel model-comparison">
          <div className="card-title">
            <BarChart3 size={18} />
            Model context
          </div>

          <div className="model-row selected">
            <div>
              <strong>
                RF V3 Expanded
              </strong>

              <span>
                Current backend classifier
              </span>
            </div>

            <div className="model-metric">
              <strong>
                88.21%
              </strong>

              <span>
                Accuracy
              </span>
            </div>

            <div className="model-metric">
              <strong>
                83.64%
              </strong>

              <span>
                Macro-F1
              </span>
            </div>
          </div>

          <div className="model-row">
            <div>
              <strong>
                ResNet-18 Fine-tuned
              </strong>

              <span>
                Deep-learning comparison
              </span>
            </div>

            <div className="model-metric">
              <strong>
                79.86%
              </strong>

              <span>
                Accuracy
              </span>
            </div>

            <div className="model-metric">
              <strong>
                74.35%
              </strong>

              <span>
                Macro-F1
              </span>
            </div>
          </div>

          <div className="selection-note">
            The expanded RF model improves
            held-out performance over the
            earlier classifier and provides
            interpretable feature-based
            predictions.
          </div>
        </div>
      </div>

      <ConfusionMatrix />
    </div>
  );
}

/* ============================================================
   DATASET METRIC
   ============================================================ */

function DatasetMetric({
  label,
  value,
}) {
  return (
    <div className="dataset-metric">
      <span>{label}</span>

      <strong>
        {Number(value).toLocaleString()}
      </strong>
    </div>
  );
}

/* ============================================================
   METRIC BAR
   ============================================================ */

function MetricBar({
  label,
  value,
}) {
  return (
    <div className="metric-bar">
      <div>
        <span>{label}</span>

        <strong>
          {value.toFixed(2)}%
        </strong>
      </div>

      <div className="bar-track">
        <div
          className="bar-fill"
          style={{
            width: `${value}%`,
          }}
        />
      </div>
    </div>
  );
}

/* ============================================================
   CONFUSION MATRIX
   ============================================================ */

function ConfusionMatrix() {
  return (
    <div className="panel confusion-panel">
      <div className="card-title">
        <Layers3 size={18} />
        Test confusion matrix
      </div>

      <p className="matrix-description">
        Rows represent the actual class and
        columns represent the predicted class.
        Diagonal values are correct
        predictions.
      </p>

      <div className="confusion-wrapper">
        <table className="confusion-matrix">
          <thead>
            <tr>
              <th>Actual / Predicted</th>

              {CLASS_NAMES.map(
                (name) => (
                  <th key={name}>
                    {name}
                  </th>
                )
              )}
            </tr>
          </thead>

          <tbody>
            {CONFUSION_MATRIX.map(
              (row, rowIndex) => (
                <tr
                  key={
                    CLASS_NAMES[
                      rowIndex
                    ]
                  }
                >
                  <th>
                    {
                      CLASS_NAMES[
                        rowIndex
                      ]
                    }
                  </th>

                  {row.map(
                    (
                      value,
                      columnIndex
                    ) => {
                      const diagonal =
                        rowIndex ===
                        columnIndex;

                      return (
                        <td
                          key={`${rowIndex}-${columnIndex}`}
                          className={
                            diagonal
                              ? "matrix-correct"
                              : ""
                          }
                        >
                          {value}
                        </td>
                      );
                    }
                  )}
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ============================================================
   LIMITATIONS
   ============================================================ */

function LimitationsSection() {
  return (
    <div className="limitations-grid">
      <div className="limitation-card warning-card">
        <AlertTriangle size={21} />

        <div>
          <h3>
            Clean-image classification is the weakest class
          </h3>

          <p>
            The <strong>none</strong> class
            achieved an F1 of{" "}
            <strong>0.54</strong>, making it
            the clearest area for further
            improvement.
          </p>
        </div>
      </div>

      <div className="limitation-card">
        <Info size={21} />

        <div>
          <h3>
            Corruption remains more difficult
          </h3>

          <p>
            Corruption achieved an F1 of{" "}
            <strong>0.78</strong>. The test
            matrix shows confusion with
            noise, blur, none and
            overexposure.
          </p>
        </div>
      </div>

      <div className="limitation-card">
        <Database size={21} />

        <div>
          <h3>
            Dataset dependence
          </h3>

          <p>
            The expanded dataset contains
            9,500 generated/processed
            examples. Images with substantially
            different real-world degradation
            patterns may behave differently.
          </p>
        </div>
      </div>

      <div className="limitation-card">
        <ShieldCheck size={21} />

        <div>
          <h3>
            Confidence is not guaranteed correctness
          </h3>

          <p>
            Classifier confidence is a model
            signal, not a guarantee that the
            predicted issue is correct.
            Human review can still be
            appropriate for important images.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   HISTORY
   ============================================================ */

function HistorySection({
  history,
  loading,
  onOpen,
}) {
  if (loading) {
    return (
      <div className="panel history-empty">
        Loading analysis history...
      </div>
    );
  }

  if (!history.length) {
    return (
      <div className="panel history-empty">
        No previous analyses found.
      </div>
    );
  }

  return (
    <div className="history-table panel">
      <div className="history-head">
        <span>IMAGE</span>
        <span>SCORE</span>
        <span>DECISION</span>
        <span>DATE</span>
      </div>

      {history.map(
        (record) => {
          const accepted =
            record.quality_label ===
              "ACCEPTABLE" ||
            record.decision ===
              "ACCEPTED";

          return (
            <button
              type="button"
              className="history-row history-row-button"
              key={record.id}
              onClick={() =>
                onOpen(record.id)
              }
            >
              <div className="history-file">
                <div className="history-file-icon">
                  <FileImage size={17} />
                </div>

                <div>
                  <strong>
                    {record.filename}
                  </strong>

                  <span>
                    Analysis #{record.id}
                  </span>
                </div>
              </div>

              <strong>
                {record.quality_score}/100
              </strong>

              <span
                className={`history-decision ${
                  accepted
                    ? "accepted"
                    : "rejected"
                }`}
              >
                {record.decision ||
                  (accepted
                    ? "ACCEPTED"
                    : "REVIEW")}
              </span>

              <span className="history-date">
                <Clock3 size={14} />
                {formatDate(
                  record.created_at
                )}
              </span>
            </button>
          );
        }
      )}
    </div>
  );
}

/* ============================================================
   HISTORY MODAL
   ============================================================ */

function HistoryModal({
  record,
  loading,
  onClose,
}) {
  return (
    <div
      className="modal-overlay"
      onMouseDown={(event) => {
        if (
          event.target ===
          event.currentTarget
        ) {
          onClose();
        }
      }}
    >
      <div className="history-modal">
        <div className="modal-header">
          <div>
            <div className="eyebrow">
              SAVED ANALYSIS
            </div>

            <h2>
              {loading
                ? "Loading..."
                : `Analysis #${record?.id}`}
            </h2>
          </div>

          <button
            className="remove-button"
            onClick={onClose}
            aria-label="Close analysis details"
          >
            <XCircle size={20} />
          </button>
        </div>

        {loading ? (
          <div className="modal-loading">
            <span className="spinner" />
            Loading analysis details...
          </div>
        ) : record ? (
          <HistoryDetail record={record} />
        ) : null}
      </div>
    </div>
  );
}

/* ============================================================
   HISTORY DETAIL
   ============================================================ */

function HistoryDetail({
  record,
}) {
  const issue =
    record.issues?.[0];

  const accepted =
    record.decision ===
    "ACCEPTED";

  return (
    <div className="history-detail">
      <div className="history-detail-file">
        <FileImage size={18} />

        <div>
          <strong>
            {record.filename}
          </strong>

          <span>
            {formatDate(
              record.created_at
            )}
          </span>
        </div>
      </div>

      <div className="history-detail-score">
        <div>
          <span>Quality score</span>

          <strong>
            {record.quality_score}/100
          </strong>
        </div>

        <span
          className={`history-decision ${
            accepted
              ? "accepted"
              : "rejected"
          }`}
        >
          {record.decision ||
            "REVIEW"}
        </span>
      </div>

      <div className="history-detail-grid">
        <DetailItem
          label="Quality label"
          value={
            record.quality_label ||
            "—"
          }
        />

        <DetailItem
          label="Detected issue"
          value={
            issue
              ? capitalize(
                  issue.type
                )
              : "None"
          }
        />

        <DetailItem
          label="Severity"
          value={
            issue
              ? capitalize(
                  issue.severity
                )
              : "None"
          }
        />

        <DetailItem
          label="Confidence"
          value={
            issue
              ? `${(
                  issue.confidence *
                  100
                ).toFixed(1)}%`
              : "—"
          }
        />
      </div>

      {issue && (
        <div className="modal-section">
          <div className="card-title">
            <Brain size={17} />
            Classification
          </div>

          <p>
            The stored result identified{" "}
            <strong>
              {capitalize(
                issue.type
              )}
            </strong>{" "}
            with{" "}
            <strong>
              {(
                issue.confidence *
                100
              ).toFixed(1)}
              %
            </strong>{" "}
            confidence and{" "}
            <strong>
              {capitalize(
                issue.severity
              )}
            </strong>{" "}
            severity.
          </p>
        </div>
      )}
    </div>
  );
}

/* ============================================================
   DETAIL ITEM
   ============================================================ */

function DetailItem({
  label,
  value,
}) {
  return (
    <div className="detail-item">
      <span>{label}</span>

      <strong>{value}</strong>
    </div>
  );
}

/* ============================================================
   ABOUT
   ============================================================ */

function AboutSection({
  modelInfo,
}) {
  const backendModel =
    modelInfo?.model ||
    "Random Forest V3 Expanded";

  return (
    <div className="about-grid">
      <div className="about-main">
        <div className="eyebrow">
          ABOUT THE SYSTEM
        </div>

        <h2>
          Why Random Forest V3 Expanded?
        </h2>

        <p>
          Image quality degradation can
          be described through measurable
          visual characteristics such as
          sharpness, brightness, contrast,
          clipping, noise, edge density and
          blockiness.
        </p>

        <p>
          The expanded Random Forest model
          was trained using 9,500 examples
          and evaluated on a held-out test
          set of 1,425 images. It provides
          issue classification using
          interpretable computer-vision
          features.
        </p>

        <p>
          Isolation Forest is used as a
          separate anomaly detector. It
          provides an additional signal
          about unusual feature patterns
          and does not replace the primary
          classifier.
        </p>

        <div className="about-pills">
          <span>
            Computer Vision
          </span>

          <span>
            Random Forest V3 Expanded
          </span>

          <span>
            Isolation Forest
          </span>

          <span>SQLite</span>

          <span>FastAPI</span>
        </div>
      </div>

      <div className="about-side">
        <div className="card-title">
          <Info size={18} />
          Terms & limitations
        </div>

        <ul>
          <li>
            The quality score is a
            system-defined quality index,
            not a universal image-quality
            standard.
          </li>

          <li>
            Model performance is based on
            the held-out evaluation dataset.
          </li>

          <li>
            Predictions can be uncertain,
            particularly for visually
            overlapping degradation types.
          </li>

          <li>
            Confidence is a model signal,
            not a guarantee of correctness.
          </li>

          <li>
            Automated results should be
            treated as decision-support
            information.
          </li>
        </ul>

        <div className="system-version">
          <span>
            Backend model
          </span>

          <strong>
            {backendModel}
          </strong>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   HELPERS
   ============================================================ */

function formatBytes(bytes) {
  if (!bytes) {
    return "0 KB";
  }

  return `${(
    bytes / 1024
  ).toFixed(1)} KB`;
}

function formatStat(value) {
  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return "—";
  }

  return numeric.toFixed(3);
}

function formatDate(value) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return value;
  }

  return date.toLocaleString();
}

function capitalize(value) {
  if (!value) {
    return "Unknown";
  }

  return (
    value.charAt(0).toUpperCase() +
    value.slice(1)
  );
}

export default App;