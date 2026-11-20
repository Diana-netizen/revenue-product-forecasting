
import { useEffect, useRef, useMemo } from "react";
import ReactECharts from "echarts-for-react";

export default function RevenueChart({ forecasts = [], target = "Revenue" }) {
  const chartRef = useRef(null);

  useEffect(() => {
    const chart = chartRef.current?.getEchartsInstance();
    const resize = () => chart?.resize();

    window.addEventListener("resize", resize);

    return () => window.removeEventListener("resize", resize);
  }, []);

  const option = useMemo(() => {
    const dates = forecasts.map((item) => item.date);

    // Revenue is displayed in KES millions.
    // Quantity retains its original units.
    const divisor = target === "Revenue" ? 1000000 : 1;

    const lower = forecasts.map((item) => item.lower / divisor);
    const median = forecasts.map((item) => item.median / divisor);
    const upper = forecasts.map((item) => item.upper / divisor);

    const confidenceRange = upper.map(
      (value, index) => value - lower[index]
    );

    return {
      animationDuration: 600,

      grid: {
        left: 65,
        right: 25,
        top: 45,
        bottom: 65,
      },

      legend: {
        top: 0,
        right: 0,
        textStyle: { color: "#778399", fontSize: 10 },
        itemWidth: 14,
        itemHeight: 3,
      },

      tooltip: {
        trigger: "axis",
        backgroundColor: "#172238",
        borderWidth: 0,
        textStyle: { color: "#fff", fontSize: 11 },
        axisPointer: { type: "line" },
        formatter: (params) => {
          const index = params[0]?.dataIndex;
          const point = forecasts[index];

          if (!point) return "";

          const format = (value) =>
            target === "Revenue"
              ? `KES ${Number(value).toLocaleString("en-KE", {
                  maximumFractionDigits: 0,
                })}`
              : Number(value).toLocaleString("en-KE", {
                  maximumFractionDigits: 2,
                });

          return `
            <strong>${point.date}</strong><br/>
            P10: ${format(point.lower)}<br/>
            P50: ${format(point.median)}<br/>
            P90: ${format(point.upper)}
          `;
        },
      },

      xAxis: {
        type: "category",
        boundaryGap: false,
        data: dates,
        axisLine: { lineStyle: { color: "#e8edf4" } },
        axisTick: { show: false },
        axisLabel: {
          color: "#8993a5",
          fontSize: 10,
          rotate: 35,
          formatter: (value) => value.slice(5),
        },
      },

      yAxis: {
        type: "value",
        name:
          target === "Revenue"
            ? "Revenue (KES millions)"
            : "Quantity",
        nameTextStyle: { color: "#8993a5", fontSize: 10 },
        splitLine: {
          lineStyle: { color: "#edf0f5", type: "dashed" },
        },
        axisLabel: {
          color: "#8993a5",
          fontSize: 10,
        },
      },

      series: [
        {
          name: "P10",
          type: "line",
          data: lower,
          stack: "confidence",
          stackStrategy: "all",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
          tooltip: { show: false },
        },
        {
          name: "P90",
          type: "line",
          data: confidenceRange,
          stack: "confidence",
          stackStrategy: "all",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(22,160,133,0.16)" },
          tooltip: { show: false },
        },
        {
          name: "Forecast P50",
          type: "line",
          data: median,
          smooth: 0.2,
          symbol: "none",
          lineStyle: { width: 2.5, color: "#16a085" },
          itemStyle: { color: "#16a085" },
        },
      ],
    };
  }, [forecasts, target]);

  if (!forecasts.length) {
    return (
      <div className="revenue-chart">
        <div
          style={{
            height: "340px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#8993a5",
          }}
        >
          Generate a forecast to display results.
        </div>
      </div>
    );
  }

  return (
    <div className="revenue-chart">
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: "340px", width: "100%" }}
        opts={{ renderer: "canvas" }}
        notMerge={true}
        lazyUpdate={true}
      />
      <p className="chart-disclaimer">
        Actual model-generated forecasts. Shaded area represents the P10–P90
        prediction interval.
      </p>
    </div>
  );
}
