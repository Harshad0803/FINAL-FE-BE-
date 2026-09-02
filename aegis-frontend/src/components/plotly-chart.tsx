import React, { useEffect, useRef } from "react";
import plotlyConfig from "@/utils/plotlyConfig";

interface PlotlyChartProps {
  figure: any;
  useContainerWidth?: boolean;
  style?: React.CSSProperties;
  config?: Record<string, any>;
}

const PlotlyChart: React.FC<PlotlyChartProps> = ({ figure, useContainerWidth = true, style, config }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotlyModuleRef = useRef<any>(null);

  useEffect(() => {
    let mounted = true;

    const renderPlot = async () => {
      if (!plotlyModuleRef.current) {
        const mod: any = await import("plotly.js-basic-dist");
        if (!mounted) return;
        // plotly.js-basic-dist is a CJS/UMD bundle (module.exports = Plotly).
        // Dynamic import() of a CJS module surfaces its exports under
        // `.default`; the top-level named properties (newPlot, react, etc.)
        // aren't reliably re-exported for a bundle this size since Plotly
        // assigns most of its API at runtime rather than via static exports.
        // Falling back to `mod` keeps this working if a bundler ever does
        // flatten it.
        plotlyModuleRef.current = mod.default ?? mod;
      }
      const Plotly = plotlyModuleRef.current;
      if (!Plotly?.react) {
        console.error("PlotlyChart: Plotly.react is unavailable — module shape was", Plotly);
        return;
      }
      if (containerRef.current && mounted) {
        try {
          // Pass the shared Plotly config so every chart uses the same ModeBar
          // and interactivity settings. Do not modify figure.data or figure.layout.
          const mergedConfig = { ...plotlyConfig, ...(config ?? {}) };
          // Plotly.react diffs against whatever is already drawn (and just
          // calls newPlot itself the first time) instead of tearing the
          // plot down and rebuilding it from scratch, so real data changes
          // (a re-trained model, a different chart tab) redraw smoothly
          // instead of flashing to a blank canvas and back.
          await Plotly.react(containerRef.current, figure.data, figure.layout, mergedConfig);
        } catch (err) {
          console.error("PlotlyChart: failed to render figure", err);
        }
      }
    };

    renderPlot();

    return () => {
      mounted = false;
    };
  }, [figure, config]);

  // Purge only on true unmount (not on every figure/config update above —
  // that would defeat the point of using `react` for smooth in-place
  // updates).
  useEffect(() => {
    return () => {
      const Plotly = plotlyModuleRef.current;
      if (Plotly?.purge && containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, []);

  const containerStyle: React.CSSProperties = style?.height
    ? { width: "100%", ...style }
    : { width: "100%", minHeight: 320, ...style };

  return <div ref={containerRef} style={containerStyle} />;
};

export default PlotlyChart;
