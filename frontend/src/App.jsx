
import { useState } from "react";
import RevenueChart from "./components/RevenueChart";
import BackendStatus from "./components/BackendStatus";

import {
  LayoutDashboard,
  TrendingUp,
  BarChart3,
  Boxes,
  Settings,
  Bell,
  Search,
  CalendarDays,
  Menu,
  X,
  ArrowUpRight,
  Activity,
  Download,
  ChevronDown,
  CircleHelp,
} from "lucide-react";

const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Forecasting", icon: TrendingUp },
  { label: "Product Analysis", icon: Boxes },
  { label: "Model Performance", icon: BarChart3 },
];

const PRODUCTS = [
  "Product 1",
  "Product 2",
  "Product 3",
  "Product 4",
];

const FORECAST_ORIGIN = "2026-10-01";

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Forecast configuration
  const [selectedProduct, setSelectedProduct] = useState("Product 1");
  const [selectedModel, setSelectedModel] = useState("lgbm");
  const [selectedTarget, setSelectedTarget] = useState("Revenue");
  const [forecastHorizon, setForecastHorizon] = useState(30);

  // API state
  const [forecastData, setForecastData] = useState([]);
  const [requestStatus, setRequestStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [hasGeneratedForecast, setHasGeneratedForecast] = useState(false);

  // Generate forecast through FastAPI
  const handleGenerateForecast = async () => {
    setLoading(true);
    setRequestStatus("Generating forecast...");
    setForecastData([]);
    setHasGeneratedForecast(false);

    const payload = {
      product: selectedProduct,
      model: selectedModel,
      target: selectedTarget,
      horizon: forecastHorizon,
      forecast_origin: FORECAST_ORIGIN,
    };

    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Forecast request failed."
        );
      }

      setForecastData(data.forecasts);
      setHasGeneratedForecast(true);

      setRequestStatus(
        `Successfully generated ${data.horizon} daily forecasts for ${data.product}.`
      );
    } catch (error) {
      setForecastData([]);
      setHasGeneratedForecast(false);
      setRequestStatus(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Calculate the total forecast from the returned median values.
  const totalForecast = forecastData.reduce(
    (total, point) => total + Number(point.median || 0),
    0
  );

  const forecastMetric =
    hasGeneratedForecast && forecastData.length > 0
      ? selectedTarget === "Revenue"
        ? `KES ${(totalForecast / 1000000).toFixed(2)}M`
        : totalForecast.toLocaleString("en-KE", {
            maximumFractionDigits: 0,
          })
      : "—";

  const metrics = [
    {
      label: "Historical Revenue",
      value: "Not connected",
      change: "Historical data integration pending",
      icon: BarChart3,
    },
    {
      label:
        selectedTarget === "Revenue"
          ? "Forecast Revenue"
          : "Forecast Quantity",
      value: forecastMetric,
      change: hasGeneratedForecast
        ? `${forecastData.length} daily forecast values · P50 total`
        : "Generate a forecast to view",
      icon: TrendingUp,
    },
    {
      label: "Forecast Accuracy",
      value: "—",
      change: "Based on final test evaluation",
      icon: Activity,
    },
    {
      label: "Products Monitored",
      value: "4",
      change: "Configured products",
      icon: Boxes,
    },
  ];

  return (
    <div className="app-shell">

      {/* Mobile navigation overlay */}
      {sidebarOpen && (
        <button
          className="mobile-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`sidebar ${
          sidebarOpen ? "sidebar-open" : ""
        }`}
      >
        <div className="brand">
          <div className="brand-icon">
            <Activity size={23} />
          </div>

          <div>
            <h2>ForecastIQ</h2>
            <span>REVENUE INTELLIGENCE</span>
          </div>

          <button
            className="mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={20} />
          </button>
        </div>

        <div className="workspace-label">
          WORKSPACE
        </div>

        <nav className="navigation">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${
                activePage === label ? "nav-active" : ""
              }`}
              onClick={() => {
                setActivePage(label);
                setSidebarOpen(false);
              }}
            >
              <Icon size={19} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="workspace-label">
            PREFERENCES
          </div>

          <button className="nav-item">
            <Settings size={19} />
            <span>Settings</span>
          </button>

          <button className="nav-item">
            <CircleHelp size={19} />
            <span>Help & Support</span>
          </button>

          <div className="sidebar-profile">
            <div className="profile-avatar">FA</div>

            <div className="profile-details">
              <strong>Forecast Analyst</strong>
              <span>Administrator</span>
            </div>

            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">

        {/* Top navigation */}
        <header className="topbar">
          <button
            className="menu-button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={22} />
          </button>

          <BackendStatus />

          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-divider">/</span>
            <strong>{activePage}</strong>
          </div>

          <div className="topbar-actions">
            <button
              className="icon-button"
              aria-label="Search"
            >
              <Search size={19} />
            </button>

            <button
              className="icon-button notification-button"
              aria-label="Notifications"
            >
              <Bell size={19} />
              <span />
            </button>

            <div className="topbar-divider" />

            <div className="topbar-date">
              <CalendarDays size={17} />
              <span>Forecast workspace</span>
            </div>
          </div>
        </header>

        <div className="dashboard-content">

          {/* Welcome section */}
          <section className="welcome-section">
            <div>
              <div className="eyebrow">
                ANALYTICS WORKSPACE
              </div>

              <h1>{activePage}</h1>

              <p>
                Monitor revenue trends, explore forecasts and
                evaluate predictive model performance.
              </p>
            </div>

            <button
              className="export-button"
              disabled={!hasGeneratedForecast}
              onClick={() => {
                const csv = [
                  ["Date", "P10", "P50", "P90"],
                  ...forecastData.map((point) => [
                    point.date,
                    point.lower,
                    point.median,
                    point.upper,
                  ]),
                ]
                  .map((row) => row.join(","))
                  .join("\n");

                const blob = new Blob([csv], {
                  type: "text/csv;charset=utf-8;",
                });

                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");

                link.href = url;
                link.download = `${selectedProduct}_${selectedTarget}_forecast.csv`;
                link.click();

                URL.revokeObjectURL(url);
              }}
            >
              <Download size={17} />
              Export forecast
            </button>
          </section>

          {/* Reporting context */}
          <section className="filter-bar">
            <div className="filter-context">
              <span className="filter-label">
                REPORTING VIEW
              </span>
              <strong>{selectedProduct}</strong>
            </div>

            <div className="filter-context">
              <span className="filter-label">
                TARGET
              </span>
              <strong>{selectedTarget}</strong>
            </div>

            <div className="filter-context">
              <span className="filter-label">
                PERIOD
              </span>
              <strong>
                {hasGeneratedForecast
                  ? `${forecastData.length} forecast days`
                  : "Awaiting forecast"}
              </strong>
            </div>

            <div className="filter-status">
              <span className="status-dot" />
              {hasGeneratedForecast
                ? "Forecast available"
                : "Ready for forecasting"}
            </div>
          </section>

          {/* Forecast configuration */}
          <section className="panel configuration-panel">
            <div className="panel-heading">
              <div>
                <h2>Forecast configuration</h2>
                <p>
                  Configure the parameters for your forecast.
                </p>
              </div>
            </div>

            <div className="forecast-controls">

              {/* Product */}
              <div className="control-group">
                <label htmlFor="product">
                  Product
                </label>

                <select
                  id="product"
                  value={selectedProduct}
                  onChange={(e) => {
                    setSelectedProduct(e.target.value);
                    setHasGeneratedForecast(false);
                    setForecastData([]);
                  }}
                >
                  {PRODUCTS.map((product) => (
                    <option
                      key={product}
                      value={product}
                    >
                      {product}
                    </option>
                  ))}
                </select>
              </div>

              {/* Model */}
              <div className="control-group">
                <label htmlFor="model">
                  Forecasting model
                </label>

                <select
                  id="model"
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    setHasGeneratedForecast(false);
                    setForecastData([]);
                  }}
                >
                  <option value="lgbm">
                    LightGBM
                  </option>

                  <option value="gru" disabled>
                    GRU (not deployed)
                  </option>

                  <option value="tft" disabled>
                    Temporal Fusion Transformer (not deployed)
                  </option>
                </select>
              </div>

              {/* Forecast target */}
              <div className="control-group">
                <label htmlFor="target">
                  Forecast target
                </label>

                <select
                  id="target"
                  value={selectedTarget}
                  onChange={(e) => {
                    setSelectedTarget(e.target.value);
                    setHasGeneratedForecast(false);
                    setForecastData([]);
                  }}
                >
                  <option value="Revenue">
                    Revenue
                  </option>

                  <option value="Quantity">
                    Quantity
                  </option>
                </select>
              </div>

              {/* Forecast horizon */}
              <div className="control-group">
                <label htmlFor="horizon">
                  Forecast horizon
                </label>

                <select
                  id="horizon"
                  value={forecastHorizon}
                  onChange={(e) => {
                    setForecastHorizon(Number(e.target.value));
                    setHasGeneratedForecast(false);
                    setForecastData([]);
                  }}
                >
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                  <option value={180}>180 days</option>
                  <option value={365}>365 days</option>
                </select>
              </div>

            </div>

            {/* Forecast origin */}
            <div className="control-group">
              <label htmlFor="forecastOrigin">
                Forecast origin
              </label>

              <input
                id="forecastOrigin"
                type="date"
                value={FORECAST_ORIGIN}
                disabled
              />
            </div>

            {/* Generate button */}
            <div className="configuration-footer">
              <span>
                Quantile predictions: P10, P50 and P90
              </span>

              <button
                className="generate-button"
                onClick={handleGenerateForecast}
                disabled={loading}
              >
                {loading
                  ? "Generating..."
                  : "Generate forecast"}
              </button>
            </div>

            {requestStatus && (
              <p
                className="request-status"
                role="status"
                aria-live="polite"
              >
                {requestStatus}
              </p>
            )}
          </section>

          {/* Performance metrics */}
          <section className="section-heading">
            <div>
              <h2>Performance overview</h2>
              <p>
                Your key forecasting indicators at a glance.
              </p>
            </div>
          </section>

          <section className="metrics-grid">
            {metrics.map(
              ({ label, value, change, icon: Icon }) => (
                <article
                  className="metric-card"
                  key={label}
                >
                  <div className="metric-top">
                    <span>{label}</span>

                    <div className="metric-icon">
                      <Icon size={19} />
                    </div>
                  </div>

                  <div className="metric-value">
                    {value}
                  </div>

                  <div className="metric-footnote">
                    <span className="metric-neutral">
                      {change}
                    </span>
                  </div>
                </article>
              )
            )}
          </section>

          {/* Forecast chart and model information */}
          <section className="analysis-grid">

            <article className="panel forecast-panel">
              <div className="panel-heading">
                <div>
                  <h2>
                    {selectedTarget} forecast
                  </h2>

                  <p>
                    {selectedProduct} · LightGBM · P10, P50 and P90
                  </p>
                </div>

                <button
                  className="select-button"
                  disabled
                >
                  {selectedProduct}
                  <ChevronDown size={15} />
                </button>
              </div>

              <div className="chart-container">
                <RevenueChart
                  forecasts={forecastData}
                  target={selectedTarget}
                />
              </div>
            </article>

            {/* Model availability */}
            <article className="panel model-panel">
              <div className="panel-heading">
                <div>
                  <h2>Forecasting models</h2>
                  <p>
                    Model deployment status
                  </p>
                </div>
              </div>

              <div className="model-list">
                {[
                  {
                    name: "LightGBM",
                    description: "Gradient boosting",
                    status: "Deployed",
                    active: true,
                  },
                  {
                    name: "GRU",
                    description: "Recurrent neural network",
                    status: "Not deployed",
                    active: false,
                  },
                  {
                    name: "TFT",
                    description: "Temporal Fusion Transformer",
                    status: "Not deployed",
                    active: false,
                  },
                ].map(
                  ({ name, description, status, active }) => (
                    <div
                      className="model-row"
                      key={name}
                    >
                      <div className="model-symbol">
                        {name.slice(0, 1)}
                      </div>

                      <div className="model-description">
                        <strong>{name}</strong>
                        <span>{description}</span>
                      </div>

                      <span
                        className={
                          active
                            ? "model-active"
                            : "model-pending"
                        }
                      >
                        {status}
                      </span>
                    </div>
                  )
                )}
              </div>

              <div className="model-note">
                LightGBM is currently serving the saved
                production forecasts.
              </div>
            </article>
          </section>

          {/* Product coverage */}
          <section className="panel activity-panel">
            <div className="panel-heading">
              <div>
                <h2>Product performance</h2>
                <p>
                  Forecasting coverage across your publications
                </p>
              </div>

              <button
                className="view-button"
                disabled
              >
                View analysis
                <ArrowUpRight size={15} />
              </button>
            </div>

            <div className="product-grid">
              {PRODUCTS.map((product, index) => (
                <div
                  className="product-card"
                  key={product}
                >
                  <div className="product-number">
                    0{index + 1}
                  </div>

                  <strong>{product}</strong>

                  <span>
                    {hasGeneratedForecast &&
                    selectedProduct === product
                      ? `${forecastData.length} forecast days available`
                      : "Select product to generate forecast"}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* Footer */}
          <footer className="dashboard-footer">
            <span>
              ForecastIQ · Revenue Forecasting System
            </span>

            <span>
              Forecasting research workspace
            </span>
          </footer>

        </div>
      </main>
    </div>
  );
}

export default App;
