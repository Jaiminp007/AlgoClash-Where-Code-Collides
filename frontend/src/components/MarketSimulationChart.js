import { useMemo, useRef } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart
} from 'recharts';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import './MarketSimulationChart.css';

const MarketSimulationChart = ({ chartData, hideAgentCards = false }) => {
  const chartRef = useRef(null);

  // Agent color palette - vibrant colors for visibility
  const agentColors = [
    '#FF6B6B', // Red
    '#4ECDC4', // Teal
    '#FFE66D', // Yellow
    '#A8E6CF', // Mint Green
    '#FF8B94', // Pink
    '#C7CEEA', // Lavender
    '#95E1D3', // Aqua
    '#F38181', // Coral
    '#AA96DA', // Purple
    '#FCBAD3', // Rose
  ];

  const processedData = useMemo(() => {
    if (!chartData || chartData.length === 0) return { chartPoints: [], agents: [], agentColorMap: {}, priceAxisId: 'price', portfolioAxisId: 'portfolio' };

    // Extract all unique agents from portfolio data
    const agentSet = new Set();
    chartData.forEach(tick => {
      if (tick.agent_portfolios) {
        Object.keys(tick.agent_portfolios).forEach(agent => {
          agentSet.add(agent);
        });
      }
    });

    const agents = Array.from(agentSet).filter(agent => !agent.startsWith('Liquidity_'));
    const agentColorMap = {};
    agents.forEach((agent, idx) => {
      agentColorMap[agent] = agentColors[idx % agentColors.length];
    });

    // Process each tick to build chart data
    const chartPoints = chartData.map(tick => {
      const point = {
        tick: tick.tick,
        price: tick.price,
        timestamp: tick.timestamp,
        trades: tick.trades || []
      };

      // Add portfolio value for each agent
      if (tick.agent_portfolios) {
        agents.forEach(agent => {
          if (tick.agent_portfolios[agent]) {
            point[`${agent}_value`] = tick.agent_portfolios[agent].value;
            point[`${agent}_cash`] = tick.agent_portfolios[agent].cash;
            point[`${agent}_stock`] = tick.agent_portfolios[agent].stock;
          }
        });
      }

      return point;
    });

    return { chartPoints, agents, agentColorMap, priceAxisId: 'price', portfolioAxisId: 'portfolio' };
  }, [chartData]);

  if (!chartData || chartData.length === 0) {
    return (
      <div className="chart-placeholder">
        <div className="chart-loading">
          <div className="chart-spinner"></div>
          <p>Waiting for market simulation to start...</p>
        </div>
      </div>
    );
  }

  const { chartPoints, agents, agentColorMap, priceAxisId, portfolioAxisId } = processedData;

  // Calculate price range for left Y-axis
  const prices = chartPoints.map(d => d.price).filter(p => p > 0);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = maxPrice - minPrice;
  const priceYAxisMin = Math.floor(minPrice - priceRange * 0.1);
  const priceYAxisMax = Math.ceil(maxPrice + priceRange * 0.1);

  // Calculate portfolio value range for right Y-axis
  const allPortfolioValues = [];
  chartPoints.forEach(point => {
    agents.forEach(agent => {
      const value = point[`${agent}_value`];
      if (value !== undefined && value > 0) {
        allPortfolioValues.push(value);
      }
    });
  });

  const minPortfolio = Math.min(...allPortfolioValues, 10000);
  const maxPortfolio = Math.max(...allPortfolioValues, 10000);
  const portfolioRange = maxPortfolio - minPortfolio;
  const portfolioYAxisMin = Math.floor(minPortfolio - portfolioRange * 0.1);
  const portfolioYAxisMax = Math.ceil(maxPortfolio + portfolioRange * 0.1);

  // Custom tooltip to show agent details
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;

      return (
        <div className="custom-tooltip">
          <p className="tooltip-tick"><strong>Tick {data.tick}</strong></p>
          <p className="tooltip-price"><strong>Market Price: ${data.price?.toFixed(2)}</strong></p>
          <div className="tooltip-divider"></div>

          {agents.map(agent => {
            const value = data[`${agent}_value`];
            const cash = data[`${agent}_cash`];
            const stock = data[`${agent}_stock`];

            if (value !== undefined) {
              const initialValue = 10000; // Initial portfolio value
              const roi = ((value - initialValue) / initialValue * 100).toFixed(2);
              const roiColor = roi >= 0 ? '#4CAF50' : '#f44336';

              return (
                <div key={agent} className="agent-tooltip-group">
                  <p className="agent-name" style={{ color: agentColorMap[agent], fontWeight: 'bold' }}>
                    {agent}
                  </p>
                  <p className="tooltip-detail">  Value: ${value.toFixed(2)}</p>
                  <p className="tooltip-detail" style={{ color: roiColor }}>
                    ROI: {roi >= 0 ? '+' : ''}{roi}%
                  </p>
                  <p className="tooltip-detail-small">
                    Cash: ${cash?.toFixed(2)} | Stock: {stock}
                  </p>
                </div>
              );
            }
            return null;
          })}
        </div>
      );
    }
    return null;
  };

  // Custom legend
  const renderLegend = () => {
    return (
      <div className="chart-legend-multi">
        <div className="legend-item-price-thick">
          <span className="legend-line-black-thick"></span> Market Price
        </div>
        {agents.map(agent => (
          <div key={agent} className="legend-item-agent-line">
            <span
              className="legend-line-agent"
              style={{ backgroundColor: agentColorMap[agent] }}
            ></span>
            {agent}
          </div>
        ))}
      </div>
    );
  };

  // Get latest tick data for agent stats
  const latestTick = chartPoints.length > 0 ? chartPoints[chartPoints.length - 1] : null;
  const displayAgents = agents.slice(0, 6); // Show only first 6 agents

  // Download chart as PDF (graph only)
  const downloadChart = async () => {
    if (!chartRef.current) return;

    try {
      // Find only the chart section (not the stats cards)
      const chartSection = chartRef.current.querySelector('.chart-section');
      if (!chartSection) return;

      // Hide the chart-stats-enhanced temporarily for cleaner export
      const statsElement = chartSection.querySelector('.chart-stats-enhanced');
      const originalStatsDisplay = statsElement ? statsElement.style.display : '';
      if (statsElement) statsElement.style.display = 'none';

      const canvas = await html2canvas(chartSection, {
        backgroundColor: '#1a1a2e',
        scale: 2, // Higher quality
        logging: false,
        useCORS: true
      });

      // Restore stats element
      if (statsElement) statsElement.style.display = originalStatsDisplay;

      // Convert canvas to PDF
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'px',
        format: [canvas.width, canvas.height]
      });

      pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      pdf.save(`market-simulation-${timestamp}.pdf`);
    } catch (error) {
      console.error('Error downloading chart:', error);
      alert('Failed to download chart. Please try again.');
    }
  };

  // Render agent stat cards
  const renderAgentStatCards = () => {
    if (!latestTick || displayAgents.length === 0) return null;

    return (
      <div className="agent-stats-grid">
        {displayAgents.map((agent) => {
          const value = latestTick[`${agent}_value`] || 10000;
          const cash = latestTick[`${agent}_cash`] || 10000;
          const stock = latestTick[`${agent}_stock`] || 0;
          const initialValue = 10000;
          const roi = ((value - initialValue) / initialValue * 100).toFixed(2);
          const roiColor = roi >= 0 ? '#4CAF50' : '#f44336';

          return (
            <div key={agent} className="agent-stat-card" style={{ borderLeft: `4px solid ${agentColorMap[agent]}` }}>
              <div className="agent-stat-header">
                <div className="agent-stat-name" style={{ color: agentColorMap[agent] }}>
                  {agent.replace('generated_algo_', '')}
                </div>
              </div>
              <div className="agent-stat-body">
                <div className="stat-row">
                  <span className="stat-label-small">ROI</span>
                  <span className="stat-value-large" style={{ color: roiColor }}>
                    {roi >= 0 ? '+' : ''}{roi}%
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label-small">Cash</span>
                  <span className="stat-value-medium">${cash.toFixed(2)}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label-small">Stock</span>
                  <span className="stat-value-medium">{stock}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <>
      <div className="chart-header">
        <div className="chart-title-section">
          <h3>📈 Live Market Simulation - Agent Performance</h3>
          <button onClick={downloadChart} className="download-chart-btn" title="Download chart as PDF">
            📄 Download PDF
          </button>
        </div>
        {renderLegend()}
      </div>

      <div ref={chartRef} className="market-simulation-layout">
        {/* Left side: Chart */}
        <div className="chart-section">
          <ResponsiveContainer width="100%" height={480}>
            <ComposedChart
              data={chartPoints}
              margin={{ top: 20, right: 80, left: 20, bottom: 30 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />

              {/* X-Axis */}
              <XAxis
                dataKey="tick"
                stroke="#ffffff"
                tick={{ fill: '#ffffff', fontSize: 12 }}
                label={{ value: 'Tick Number', position: 'insideBottom', offset: -15, fill: '#ffffff' }}
              />

              {/* Left Y-Axis (Market Price) */}
              <YAxis
                yAxisId={priceAxisId}
                orientation="left"
                domain={[priceYAxisMin, priceYAxisMax]}
                stroke="#000000"
                tick={{ fill: '#ffffff', fontSize: 12 }}
                label={{
                  value: 'Market Price ($)',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#ffffff',
                  style: { fontWeight: 'bold' }
                }}
              />

              {/* Right Y-Axis (Portfolio Value) */}
              <YAxis
                yAxisId={portfolioAxisId}
                orientation="right"
                domain={[portfolioYAxisMin, portfolioYAxisMax]}
                stroke="#4CAF50"
                tick={{ fill: '#ffffff', fontSize: 12 }}
                label={{
                  value: 'Portfolio Value ($)',
                  angle: 90,
                  position: 'insideRight',
                  fill: '#ffffff',
                  style: { fontWeight: 'bold' }
                }}
              />

              <Tooltip content={<CustomTooltip />} />

              {/* Market price line (black, thick) */}
              <Line
                yAxisId={priceAxisId}
                type="monotone"
                dataKey="price"
                stroke="#000000"
                strokeWidth={4}
                dot={false}
                name="Market Price"
                animationDuration={300}
              />

              {/* Agent portfolio value lines (colored) */}
              {agents.map(agent => (
                <Line
                  key={agent}
                  yAxisId={portfolioAxisId}
                  type="monotone"
                  dataKey={`${agent}_value`}
                  stroke={agentColorMap[agent]}
                  strokeWidth={2.5}
                  dot={false}
                  name={agent}
                  animationDuration={300}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>

          <div className="chart-stats-enhanced">
            <div className="stat-item">
              <span className="stat-label">Market Price:</span>
              <span className="stat-value">${chartPoints[chartPoints.length - 1]?.price?.toFixed(2) || '0.00'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Ticks Processed:</span>
              <span className="stat-value">{chartPoints.length}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Active Agents:</span>
              <span className="stat-value">{agents.length}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Price Range:</span>
              <span className="stat-value">${minPrice.toFixed(2)} - ${maxPrice.toFixed(2)}</span>
            </div>
            {agents.length > 0 && chartPoints.length > 0 && (
              <div className="stat-item">
                <span className="stat-label">Portfolio Range:</span>
                <span className="stat-value">
                  ${minPortfolio.toFixed(0)} - ${maxPortfolio.toFixed(0)}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Right side: Agent stat cards - hidden in results view */}
        {!hideAgentCards && (
          <div className="stats-section">
            {renderAgentStatCards()}
          </div>
        )}
      </div>
    </>
  );
};

export default MarketSimulationChart;
