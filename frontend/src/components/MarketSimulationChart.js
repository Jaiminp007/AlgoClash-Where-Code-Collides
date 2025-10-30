import React, { useMemo } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Scatter,
  ComposedChart
} from 'recharts';
import './MarketSimulationChart.css';

const MarketSimulationChart = ({ chartData }) => {
  // Agent color palette - vibrant colors for visibility
  const agentColors = [
    '#FF6B6B', // Red
    '#4ECDC4', // Teal
    '#FFE66D', // Yellow
    '#A8E6CF', // Mint
    '#FF8B94', // Pink
    '#C7CEEA', // Lavender
    '#95E1D3', // Aqua
    '#F38181', // Coral
    '#AA96DA', // Purple
    '#FCBAD3', // Rose
  ];

  const processedData = useMemo(() => {
    if (!chartData || chartData.length === 0) return { chartPoints: [], agents: [], agentColorMap: {} };

    // Extract all unique agents from trades
    const agentSet = new Set();
    chartData.forEach(tick => {
      tick.trades?.forEach(trade => {
        if (trade.agent && !trade.agent.startsWith('Liquidity_')) {
          // Clean up agent name (remove generated_algo_ prefix)
          const cleanName = trade.agent.replace('generated_algo_', '');
          agentSet.add(cleanName);
        }
      });
    });

    const agents = Array.from(agentSet);
    const agentColorMap = {};
    agents.forEach((agent, idx) => {
      agentColorMap[agent] = agentColors[idx % agentColors.length];
    });

    // Calculate price range for offset calculation
    const prices = chartData.map(d => d.price).filter(p => p > 0);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const offset = priceRange * 0.05; // 5% offset for buy/sell markers

    // Process each tick
    const chartPoints = chartData.map(tick => {
      const point = {
        tick: tick.tick,
        price: tick.price,
        timestamp: tick.timestamp,
        trades: tick.trades || []
      };

      // For each agent, create buy/sell data points
      agents.forEach(agent => {
        const agentTrades = tick.trades?.filter(t => {
          const cleanName = t.agent?.replace('generated_algo_', '');
          return cleanName === agent;
        }) || [];

        const buyTrades = agentTrades.filter(t => t.side === 'BUY' || t.side === 'buy');
        const sellTrades = agentTrades.filter(t => t.side === 'SELL' || t.side === 'sell');

        // Position buy orders above the price line
        if (buyTrades.length > 0) {
          const totalVolume = buyTrades.reduce((sum, t) => sum + (t.quantity || 0), 0);
          point[`${agent}_buy`] = tick.price + offset;
          point[`${agent}_buy_volume`] = totalVolume;
        }

        // Position sell orders below the price line
        if (sellTrades.length > 0) {
          const totalVolume = sellTrades.reduce((sum, t) => sum + (t.quantity || 0), 0);
          point[`${agent}_sell`] = tick.price - offset;
          point[`${agent}_sell_volume`] = totalVolume;
        }
      });

      return point;
    });

    return { chartPoints, agents, agentColorMap };
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

  const { chartPoints, agents, agentColorMap } = processedData;

  // Calculate price range for Y-axis
  const prices = chartPoints.map(d => d.price).filter(p => p > 0);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = maxPrice - minPrice;
  const yAxisMin = Math.floor(minPrice - priceRange * 0.15);
  const yAxisMax = Math.ceil(maxPrice + priceRange * 0.15);

  // Custom tooltip to show trade details
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;

      // Get all trades for this tick grouped by agent
      const tradesByAgent = {};
      data.trades?.forEach(trade => {
        if (!trade.agent || trade.agent.startsWith('Liquidity_')) return;
        const cleanName = trade.agent.replace('generated_algo_', '');
        if (!tradesByAgent[cleanName]) {
          tradesByAgent[cleanName] = [];
        }
        tradesByAgent[cleanName].push(trade);
      });

      return (
        <div className="custom-tooltip">
          <p className="tooltip-tick">Tick {data.tick}</p>
          <p className="tooltip-price">Market Price: ${data.price?.toFixed(2)}</p>

          {Object.keys(tradesByAgent).length > 0 && (
            <div className="tooltip-trades">
              <p className="tooltip-trades-title">Agent Activity:</p>
              {Object.entries(tradesByAgent).map(([agent, trades]) => {
                const buyVolume = trades.filter(t => t.side === 'BUY' || t.side === 'buy').reduce((sum, t) => sum + (t.quantity || 0), 0);
                const sellVolume = trades.filter(t => t.side === 'SELL' || t.side === 'sell').reduce((sum, t) => sum + (t.quantity || 0), 0);

                return (
                  <div key={agent} className="agent-trade-group">
                    <p className="agent-name" style={{ color: agentColorMap[agent] }}>
                      {agent}:
                    </p>
                    {buyVolume > 0 && (
                      <p className="tooltip-buy">  ↑ BUY: {buyVolume} shares</p>
                    )}
                    {sellVolume > 0 && (
                      <p className="tooltip-sell">  ↓ SELL: {sellVolume} shares</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  // Custom legend
  const renderLegend = () => {
    return (
      <div className="chart-legend-agents">
        <div className="legend-item-price">
          <span className="legend-line-black"></span> Market Price
        </div>
        {agents.map(agent => (
          <div key={agent} className="legend-item-agent">
            <span
              className="legend-dot-agent"
              style={{ backgroundColor: agentColorMap[agent] }}
            ></span>
            {agent}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="market-simulation-chart">
      <div className="chart-header">
        <h3>📈 Live Market Simulation</h3>
        {renderLegend()}
      </div>
      <ResponsiveContainer width="100%" height={450}>
        <ComposedChart
          data={chartPoints}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
          <XAxis
            dataKey="tick"
            stroke="#ffffff"
            tick={{ fill: '#ffffff', fontSize: 12 }}
            label={{ value: 'Tick', position: 'insideBottom', offset: -10, fill: '#ffffff' }}
          />
          <YAxis
            domain={[yAxisMin, yAxisMax]}
            stroke="#ffffff"
            tick={{ fill: '#ffffff', fontSize: 12 }}
            label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', fill: '#ffffff' }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Black price line for market price */}
          <Line
            type="monotone"
            dataKey="price"
            stroke="#000000"
            strokeWidth={3}
            dot={false}
            animationDuration={300}
          />

          {/* Scatter points for each agent's buy and sell orders */}
          {agents.map(agent => (
            <React.Fragment key={agent}>
              {/* Buy orders (above the line) */}
              <Scatter
                dataKey={`${agent}_buy`}
                fill={agentColorMap[agent]}
                shape="triangle"
                r={7}
                name={`${agent} Buy`}
              />

              {/* Sell orders (below the line) */}
              <Scatter
                dataKey={`${agent}_sell`}
                fill={agentColorMap[agent]}
                shape="triangle"
                r={7}
                name={`${agent} Sell`}
                style={{ transform: 'rotate(180deg)' }}
              />
            </React.Fragment>
          ))}
        </ComposedChart>
      </ResponsiveContainer>
      <div className="chart-stats">
        <div className="stat-item">
          <span className="stat-label">Current Price:</span>
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
      </div>
    </div>
  );
};

export default MarketSimulationChart;
