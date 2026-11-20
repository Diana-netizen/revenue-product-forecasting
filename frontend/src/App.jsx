import { useEffect, useState } from "react";
import RevenueChart from "./components/RevenueChart";
import BackendStatus from "./components/BackendStatus";

import {
  LayoutDashboard, TrendingUp, BarChart3, Boxes,
  Settings, Bell, Search, CalendarDays, Menu, X,
  ArrowUpRight, Activity, Download, ChevronDown, CircleHelp
} from "lucide-react";

const PRODUCTS = ["Product 1", "Product 2", "Product 3", "Product 4"];
const FORECAST_ORIGIN = "2026-10-01";

const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Forecasting", icon: TrendingUp },
  { label: "Product Analysis", icon: Boxes },
  { label: "Model Performance", icon: BarChart3 },
];

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [selectedProduct, setSelectedProduct] = useState("Product 1");
  const [selectedModel, setSelectedModel] = useState("lgbm");
  const [selectedTarget, setSelectedTarget] = useState("Revenue");
  const [forecastHorizon, setForecastHorizon] = useState(30);

  const [historicalData, setHistoricalData] = useState([]);
  const [accuracyData, setAccuracyData] = useState([]);
  const [modelData, setModelData] = useState([]);
  const [forecastData, setForecastData] = useState([]);

  const [loading, setLoading] = useState(false);
  const [requestStatus, setRequestStatus] = useState("");

  // Load dashboard data
  useEffect(() => {
    Promise.all([
      fetch(`/api/historical?product=${encodeURIComponent(selectedProduct)}&target=${selectedTarget}`).then(r => r.json()),
      fetch(`/api/accuracy?product=${encodeURIComponent(selectedProduct)}&target=${selectedTarget}`).then(r => r.json()),
      fetch("/api/model-comparison").then(r => r.json())
    ])
      .then(([historical, accuracy, models]) => {
        setHistoricalData(historical.data || []);
        setAccuracyData(accuracy.results || []);
        setModelData(models.models || []);
      })
      .catch(error => console.error("Dashboard loading error:", error));
  }, [selectedProduct, selectedTarget]);

  // Generate forecast
  const handleGenerateForecast = async () => {
    setLoading(true);
    setRequestStatus("Generating forecast...");

    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product: selectedProduct,
          model: selectedModel,
          target: selectedTarget,
          horizon: forecastHorizon,
          forecast_origin: FORECAST_ORIGIN,
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail);

      setForecastData(data.forecasts || []);
      setRequestStatus("Forecast generated successfully.");
    } catch (error) {
      setForecastData([]);
      setRequestStatus(error.message || "Forecast failed.");
    } finally {
      setLoading(false);
    }
  };

  const historicalTotal = historicalData.reduce(
    (sum, row) => sum + Number(row.value || 0), 0
  );

  const forecastTotal = forecastData.reduce(
    (sum, row) => sum + (Number(row.median ?? row.P50 ?? row.p50) || 0), 0);

  // Find the LightGBM accuracy result for the selected target
  const maseRow = accuracyData.find(row =>
    String(row.Model || "").toLowerCase() === "lightgbm" &&
    String(row.Target || "").toLowerCase() === selectedTarget.toLowerCase() &&
    row.MASE !== undefined &&
    row.MASE !== null
  );
  
  const metrics = [
    {
      label: selectedTarget === "Revenue" ? "Historical Revenue" : "Historical Quantity",
      value: selectedTarget === "Revenue"
        ? `KES ${(historicalTotal / 1000000).toFixed(2)}M`
        : historicalTotal.toLocaleString(),
      icon: BarChart3,
    },
    {
      label: selectedTarget === "Revenue" ? "Forecast Revenue" : "Forecast Quantity",
      value: forecastData.length
        ? selectedTarget === "Revenue"
          ? `KES ${(forecastTotal / 1000000).toFixed(2)}M`
          : forecastTotal.toLocaleString()
        : "—",
      icon: TrendingUp,
    },
    {
      label: "Forecast Accuracy (MASE)",
      value: maseRow ? Number(maseRow.MASE).toFixed(3) : "—",
      icon: Activity,
    },
    {
      label: "Products Monitored",
      value: PRODUCTS.length,
      icon: Boxes,
    },
  ];

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button className="mobile-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon"><Activity size={23} /></div>
          <div><h2>ForecastIQ</h2><span>REVENUE INTELLIGENCE</span></div>
          <button className="mobile-close" onClick={() => setSidebarOpen(false)}><X size={20} /></button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${activePage === label ? "nav-active" : ""}`}
              onClick={() => {
                setActivePage(label);
                setSidebarOpen(false);
              }}
            >
              <Icon size={19} /><span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="workspace-label">PREFERENCES</div>
          <button className="nav-item"><Settings size={19} /><span>Settings</span></button>
          <button className="nav-item"><CircleHelp size={19} /><span>Help & Support</span></button>
          <div className="sidebar-profile">
            <div className="profile-avatar">FA</div>
            <div className="profile-details"><strong>Forecast Analyst</strong><span>Administrator</span></div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button className="menu-button" onClick={() => setSidebarOpen(true)}><Menu size={22} /></button>
          <BackendStatus />
          <div className="breadcrumb">
            <span>Workspace</span><span className="breadcrumb-divider">/</span><strong>{activePage}</strong>
          </div>
          <div className="topbar-actions">
            <button className="icon-button"><Search size={19} /></button>
            <button className="icon-button"><Bell size={19} /></button>
            <div className="topbar-divider" />
            <div className="topbar-date"><CalendarDays size={17} /><span>Forecast workspace</span></div>
          </div>
        </header>

        <div className="dashboard-content">
          <section className="welcome-section">
            <div>
              <div className="eyebrow">ANALYTICS WORKSPACE</div>
              <h1>{activePage}</h1>
              <p>Monitor revenue trends, explore forecasts and evaluate model performance.</p>
            </div>
          </section>

          {/* Forecast configuration */}
          <section className="panel configuration-panel">
            <div className="panel-heading">
              <div><h2>Forecast configuration</h2><p>Configure your forecast parameters.</p></div>
            </div>

            <div className="forecast-controls">
              <div className="control-group">
                <label>Product</label>
                <select value={selectedProduct} onChange={e => {
                  setSelectedProduct(e.target.value);
                  setForecastData([]);
                }}>
                  {PRODUCTS.map(p => <option key={p}>{p}</option>)}
                </select>
              </div>

              <div className="control-group">
                <label>Model</label>
                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                  <option value="lgbm">LightGBM</option>
                  <option value="gru" disabled>GRU (not deployed)</option>
                  <option value="tft" disabled>TFT (not deployed)</option>
                </select>
              </div>

              <div className="control-group">
                <label>Target</label>
                <select value={selectedTarget} onChange={e => {
                  setSelectedTarget(e.target.value);
                  setForecastData([]);
                }}>
                  <option value="Revenue">Revenue</option>
                  <option value="Quantity">Quantity</option>
                </select>
              </div>

              <div className="control-group">
                <label>Forecast horizon</label>
                <select value={forecastHorizon} onChange={e => setForecastHorizon(Number(e.target.value))}>
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                  <option value={180}>180 days</option>
                  <option value={365}>365 days</option>
                </select>
              </div>
            </div>

            <div className="configuration-footer">
              <span>Quantile predictions: P10, P50 and P90</span>
              <button className="generate-button" onClick={handleGenerateForecast} disabled={loading}>
                {loading ? "Generating..." : "Generate forecast"}
              </button>
            </div>

            {requestStatus && <p className="request-status">{requestStatus}</p>}
          </section>

          {/* Performance overview */}
          <section className="section-heading">
            <div><h2>Performance overview</h2><p>Key forecasting indicators.</p></div>
          </section>

          <section className="metrics-grid">
            {metrics.map(({ label, value, icon: Icon }) => (
              <article className="metric-card" key={label}>
                <div className="metric-top">
                  <span>{label}</span>
                  <div className="metric-icon"><Icon size={19} /></div>
                </div>
                <div className="metric-value">{value}</div>
              </article>
            ))}
          </section>

          {/* Forecast chart */}
          <section className="analysis-grid">
            <article className="panel forecast-panel">
              <div className="panel-heading">
                <div>
                  <h2>{selectedTarget} forecast</h2>
                  <p>{selectedProduct} · LightGBM · P10, P50 and P90</p>
                </div>
              </div>

              <div className="chart-container">
                <RevenueChart forecasts={forecastData} target={selectedTarget} />
              </div>
            </article>

            {/* Model comparison */}
            <article className="panel model-panel">
              <div className="panel-heading">
                <div><h2>Forecasting models</h2><p>Deployment status</p></div>
              </div>

              <div className="model-list">
                {[
                  ["LightGBM", "Gradient boosting", "Deployed"],
                  ["GRU", "Recurrent neural network", "Not deployed"],
                  ["TFT", "Temporal Fusion Transformer", "Not deployed"],
                ].map(([name, description, status]) => (
                  <div className="model-row" key={name}>
                    <div className="model-symbol">{name[0]}</div>
                    <div className="model-description">
                      <strong>{name}</strong><span>{description}</span>
                    </div>
                    <span className={status === "Deployed" ? "model-active" : "model-pending"}>
                      {status}
                    </span>
                  </div>
                ))}
              </div>

              <div className="model-comparison">
                <h3>Test performance comparison</h3>
                <div style={{ overflowX: "auto" }}>
                  <table>
                    <thead>
                      <tr>
                        {Object.keys(modelData[0] || {}).map(key => <th key={key}>{key}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {modelData.map((row, i) => (
                        <tr key={i}>
                          {Object.keys(modelData[0]).map(key => (
                            <td key={key}>{row[key] == null ? "—" : row[key]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="model-note">
                LightGBM is deployed. GRU and TFT are included for evaluation only.
              </div>
            </article>
          </section>

          {/* Test accuracy */}
          <section className="panel activity-panel">
            <div className="panel-heading">
              <div><h2>Test accuracy results</h2><p>Evaluation results for the selected product and target.</p></div>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {Object.keys(accuracyData[0] || {}).map(key => <th key={key}>{key}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {accuracyData.map((row, i) => (
                    <tr key={i}>
                      {Object.keys(accuracyData[0]).map(key => (
                        <td key={key}>{row[key] == null ? "—" : row[key]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Historical records */}
          <section className="panel activity-panel">
            <div className="panel-heading">
              <div><h2>Historical {selectedTarget}</h2><p>{selectedProduct}</p></div>
            </div>

            <p>
              {historicalData.length.toLocaleString()} historical records loaded.
            </p>
          </section>

          <footer className="dashboard-footer">
            <span>ForecastIQ · Revenue Forecasting System</span>
            <span>Forecasting research workspace</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;