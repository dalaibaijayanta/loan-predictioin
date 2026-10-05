import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    RevolvingUtilizationOfUnsecuredLines: "",
    age: "",
    NumberOfTime30_59DaysPastDueNotWorse: "",
    DebtRatio: "",
    MonthlyIncome: "",
    NumberOfOpenCreditLinesAndLoans: "",
    NumberOfTimes90DaysLate: "",
    NumberRealEstateLoansOrLines: "",
    NumberOfTime60_89DaysPastDueNotWorse: "",
    NumberOfDependents: "",
  });

const [result, setResult] = useState(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState("");


const maxImpact =
  result?.explanations?.length > 0
    ? Math.max(
        ...result.explanations.map((item) =>
          Math.abs(item.impact)
        )
      )
    : 1;

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
  event.preventDefault();

  setLoading(true);
  setError("");
  setResult(null);

  try {
    const payload = {
      RevolvingUtilizationOfUnsecuredLines:
        Number(formData.RevolvingUtilizationOfUnsecuredLines),

      age:
        Number(formData.age),

      NumberOfTime30_59DaysPastDueNotWorse:
        Number(formData.NumberOfTime30_59DaysPastDueNotWorse),

      DebtRatio:
        Number(formData.DebtRatio),

      MonthlyIncome:
        formData.MonthlyIncome === ""
          ? null
          : Number(formData.MonthlyIncome),

      NumberOfOpenCreditLinesAndLoans:
        Number(formData.NumberOfOpenCreditLinesAndLoans),

      NumberOfTimes90DaysLate:
        Number(formData.NumberOfTimes90DaysLate),

      NumberRealEstateLoansOrLines:
        Number(formData.NumberRealEstateLoansOrLines),

      NumberOfTime60_89DaysPastDueNotWorse:
        Number(formData.NumberOfTime60_89DaysPastDueNotWorse),

      NumberOfDependents:
        formData.NumberOfDependents === ""
          ? null
          : Number(formData.NumberOfDependents),
    };

    console.log("Sending to backend:", payload);

    const response = await fetch(
      "http://127.0.0.1:8000/predict",
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify(payload),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail
          ? JSON.stringify(data.detail)
          : "Prediction request failed"
      );
    }

    console.log("Backend response:", data);

    setResult(data);

  } catch (err) {

    console.error("Prediction error:", err);

    setError(
      err.message || "Unable to connect to the prediction server."
    );

  } finally {

    setLoading(false);
  }
};

  return (
    <div className="app">

      {/* Header */}
      <header className="header">
        <div>
          <h1>Loan Risk Assessment</h1>
          <p>
            AI-powered loan default prediction and risk analysis
          </p>
        </div>

        <div className="ai-badge">
          AI Powered
        </div>
      </header>


      {/* Main Content */}
      <main className="main-container">

        {/* Applicant Form */}
        <section className="card application-card">

          <div className="section-title">
            <div className="title-icon">01</div>

            <div>
              <h2>Applicant Information</h2>
              <p>
                Enter the applicant's financial information
              </p>
            </div>
          </div>


          <form onSubmit={handleSubmit}>

            <div className="form-grid">

              {/* Age */}
              <div className="form-group">
                <label>Age</label>

                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  placeholder="e.g. 42"
                  min="18"
                  max="120"
                  required
                />
              </div>


              {/* Monthly Income */}
              <div className="form-group">
                <label>Monthly Income</label>

                <input
                  type="number"
                  name="MonthlyIncome"
                  value={formData.MonthlyIncome}
                  onChange={handleChange}
                  placeholder="e.g. 6500"
                  min="0"
                />
              </div>


              {/* Credit Utilization */}
              <div className="form-group">
                <label>Revolving Credit Utilization</label>

                <input
                  type="number"
                  name="RevolvingUtilizationOfUnsecuredLines"
                  value={formData.RevolvingUtilizationOfUnsecuredLines}
                  onChange={handleChange}
                  placeholder="e.g. 0.45"
                  min="0"
                  step="any"
                  required
                />

                <span className="hint">
                  Unsecured credit utilization ratio
                </span>
              </div>


              {/* Debt Ratio */}
              <div className="form-group">
                <label>Debt Ratio</label>

                <input
                  type="number"
                  name="DebtRatio"
                  value={formData.DebtRatio}
                  onChange={handleChange}
                  placeholder="e.g. 0.35"
                  min="0"
                  step="any"
                  required
                />
              </div>


              {/* 30-59 Days */}
              <div className="form-group">
                <label>30–59 Days Past Due</label>

                <input
                  type="number"
                  name="NumberOfTime30_59DaysPastDueNotWorse"
                  value={formData.NumberOfTime30_59DaysPastDueNotWorse}
                  onChange={handleChange}
                  placeholder="e.g. 1"
                  min="0"
                  required
                />
              </div>


              {/* 60-89 Days */}
              <div className="form-group">
                <label>60–89 Days Past Due</label>

                <input
                  type="number"
                  name="NumberOfTime60_89DaysPastDueNotWorse"
                  value={formData.NumberOfTime60_89DaysPastDueNotWorse}
                  onChange={handleChange}
                  placeholder="e.g. 0"
                  min="0"
                  required
                />
              </div>


              {/* 90+ Days */}
              <div className="form-group">
                <label>90+ Days Late</label>

                <input
                  type="number"
                  name="NumberOfTimes90DaysLate"
                  value={formData.NumberOfTimes90DaysLate}
                  onChange={handleChange}
                  placeholder="e.g. 0"
                  min="0"
                  required
                />
              </div>


              {/* Open Credit Lines */}
              <div className="form-group">
                <label>Open Credit Lines & Loans</label>

                <input
                  type="number"
                  name="NumberOfOpenCreditLinesAndLoans"
                  value={formData.NumberOfOpenCreditLinesAndLoans}
                  onChange={handleChange}
                  placeholder="e.g. 8"
                  min="0"
                  required
                />
              </div>


              {/* Real Estate */}
              <div className="form-group">
                <label>Real Estate Loans / Lines</label>

                <input
                  type="number"
                  name="NumberRealEstateLoansOrLines"
                  value={formData.NumberRealEstateLoansOrLines}
                  onChange={handleChange}
                  placeholder="e.g. 1"
                  min="0"
                  required
                />
              </div>


              {/* Dependents */}
              <div className="form-group">
                <label>Number of Dependents</label>

                <input
                  type="number"
                  name="NumberOfDependents"
                  value={formData.NumberOfDependents}
                  onChange={handleChange}
                  placeholder="e.g. 2"
                  min="0"
                />
              </div>

            </div>


            {/* Submit */}
            <div className="submit-area">

              <p>
                All information is processed by the trained ML model.
              </p>

              <button type="submit">
                Assess Loan Risk
                <span>→</span>
              </button>

            </div>

          </form>

        </section>


        {/* Temporary Result Preview */}
        <section className="card preview-card">

  <div className="section-title">
    <div className="title-icon">02</div>

    <div>
      <h2>Risk Assessment</h2>
      <p>
        AI-generated loan default prediction
      </p>
    </div>
  </div>


  {/* Loading */}
  {loading && (
    <div className="empty-result">

      <div className="loading-spinner"></div>

      <h3>Analyzing Application...</h3>

      <p>
        The machine learning model is evaluating
        the applicant's financial risk.
      </p>

    </div>
  )}


  {/* Error */}
  {!loading && error && (
    <div className="error-result">

      <div className="error-icon">
        !
      </div>

      <h3>Prediction Failed</h3>

      <p>{error}</p>

    </div>
  )}


  {/* Initial State */}
  {!loading && !error && !result && (
    <div className="empty-result">

      <div className="empty-icon">
        AI
      </div>

      <h3>Ready for Assessment</h3>

      <p>
        Enter the applicant information and click
        <strong> Assess Loan Risk </strong>
        to generate the prediction.
      </p>

    </div>
  )}


  {/* Prediction Result */}
  {!loading && !error && result && (

    <div className="result-container">

      {/* Main Risk Card */}
      <div
        className={`risk-result ${
          result.risk_level === "HIGH RISK"
            ? "high-risk"
            : "low-risk"
        }`}
      >

        <div className="risk-score">
          {result.risk_score}
        </div>

        <div className="risk-main-info">

          <span className="risk-label">
            Risk Score
          </span>

          <h3>{result.risk_level}</h3>

          <span className="score-description">
            Score range: 0–100
          </span>

        </div>

      </div>


      {/* Default Probability */}
      <div className="probability-box">

        <div>
          <span>Probability of Default</span>

          <small>
            Estimated likelihood of default
          </small>
        </div>

        <strong>
          {result.default_probability}%
        </strong>

      </div>


      {/* Probability Bar */}
      <div className="probability-meter">

        <div className="meter-header">

          <span>Risk Probability</span>

          <span>
            {result.default_probability}%
          </span>

        </div>

        <div className="meter-track">

          <div
            className={`meter-fill ${
              result.risk_level === "HIGH RISK"
                ? "meter-high"
                : "meter-low"
            }`}
            style={{
              width: `${Math.min(
                result.default_probability,
                100
              )}%`,
            }}
          ></div>

        </div>

      </div>


      {/* Explainability */}
      <div className="explanation-section">

        <h3>Why this prediction?</h3>

        <p className="explanation-subtitle">
          The strongest factors influencing the model's decision
        </p>


        <div className="explanation-list">

          {result.explanations.map((item, index) => {

            const relativeImpact =
              (Math.abs(item.impact) / maxImpact) * 100;

            return (

              <div
                className="explanation-item"
                key={index}
              >

                <div
                  className={`impact-indicator ${
                    item.direction === "increases_risk"
                      ? "risk-up"
                      : "risk-down"
                  }`}
                >
                  {item.direction === "increases_risk"
                    ? "↑"
                    : "↓"}
                </div>


                <div className="explanation-info">

                  <span>
                    {item.feature}
                  </span>

                  <small>
                    {item.direction === "increases_risk"
                      ? "Increases risk"
                      : "Decreases risk"}
                  </small>

                  <div className="impact-track">

                    <div
                      className={`impact-fill ${
                        item.direction === "increases_risk"
                          ? "impact-risk"
                          : "impact-safe"
                      }`}
                      style={{
                        width: `${relativeImpact}%`,
                      }}
                    ></div>

                  </div>

                </div>


                <div className="impact-value">

                  <span>
                    {item.impact > 0 ? "+" : ""}
                    {item.impact}
                  </span>

                  <small>
                    contribution
                  </small>

                </div>

              </div>

            );
          })}

        </div>


        <div className="shap-note">

          <strong>How to read this:</strong>

          <span>
            Positive contributions push the prediction
            toward higher risk, while negative contributions
            push it toward lower risk.
          </span>

        </div>

      </div>

    </div>

  )}

</section>

      </main>


      {/* Footer */}
      <footer>
        <p>
          Loan Default Prediction System • Machine Learning Risk Analysis
        </p>
      </footer>

    </div>
  );
}

export default App;
