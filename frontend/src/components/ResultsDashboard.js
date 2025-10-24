import React, { useState } from 'react';
import './ResultsDashboard.css';

const ResultsDashboard = ({ results, onBack }) => {
  const [downloadFormat, setDownloadFormat] = useState('json');

  if (!results || !results.leaderboard) {
    return null;
  }

  const leaderboard = results.leaderboard || [];
  const winner = results.winner || leaderboard[0];

  // Calculate aggregate metrics
  const calculateMetrics = () => {
    if (leaderboard.length === 0) return null;

    const totalInitial = leaderboard.reduce((sum, agent) => {
      const initial = agent.initial_value || 10000;
      return sum + initial;
    }, 0);

    const totalFinal = leaderboard.reduce((sum, agent) => {
      const final = agent.current_value || agent.final_value || 0;
      return sum + final;
    }, 0);

    const avgROI = leaderboard.reduce((sum, agent) => sum + (agent.roi || 0), 0) / leaderboard.length;

    return {
      totalInitial,
      totalFinal,
      totalPnL: totalFinal - totalInitial,
      avgROI
    };
  };

  const metrics = calculateMetrics();

  const downloadResults = () => {
    const data = {
      timestamp: new Date().toISOString(),
      winner: winner,
      leaderboard: leaderboard,
      metrics: metrics
    };

    let content, filename, type;

    if (downloadFormat === 'json') {
      content = JSON.stringify(data, null, 2);
      filename = `algoclash_results_${Date.now()}.json`;
      type = 'application/json';
    } else {
      // CSV format
      const headers = ['Rank', 'Agent', 'Initial Value', 'Final Value', 'ROI (%)', 'PnL'];
      const rows = leaderboard.map((agent, idx) => [
        idx + 1,
        agent.name,
        agent.initial_value || 10000,
        agent.current_value || agent.final_value || 0,
        (agent.roi || 0).toFixed(2),
        ((agent.current_value || 0) - (agent.initial_value || 10000)).toFixed(2)
      ]);

      content = [headers, ...rows].map(row => row.join(',')).join('\n');
      filename = `algoclash_results_${Date.now()}.csv`;
      type = 'text/csv';
    }

    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyMetrics = () => {
    const text = `AlgoClash Results
Winner: ${winner?.name} (ROI: ${winner?.roi?.toFixed(2)}%)
Average ROI: ${metrics.avgROI.toFixed(2)}%
Total PnL: $${metrics.totalPnL.toFixed(2)}`;

    navigator.clipboard.writeText(text);
    alert('Metrics copied to clipboard!');
  };

  return (
    <div className="results-dashboard">
      <div className="results-header">
        <h2>🏁 Battle Results</h2>
        <div className="results-actions-header">
          <select
            value={downloadFormat}
            onChange={(e) => setDownloadFormat(e.target.value)}
            className="format-select"
            aria-label="Download format"
          >
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
          </select>
          <button onClick={downloadResults} className="download-btn" title="Download results">
            💾 Download
          </button>
          <button onClick={copyMetrics} className="copy-btn" title="Copy metrics">
            📋 Copy
          </button>
        </div>
      </div>

      {/* Winner Highlight */}
      {winner && (
        <div className="winner-section">
          <div className="trophy-icon">🏆</div>
          <div className="winner-info">
            <div className="winner-label">Champion</div>
            <div className="winner-name">{winner.name}</div>
            <div className="winner-roi">
              ROI: <span className={winner.roi >= 0 ? 'positive' : 'negative'}>
                {winner.roi >= 0 ? '+' : ''}{winner.roi?.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* KPI Tiles */}
      {metrics && (
        <div className="kpi-grid">
          <div className="kpi-tile">
            <div className="kpi-label">Total Initial Capital</div>
            <div className="kpi-value">${metrics.totalInitial.toLocaleString()}</div>
          </div>
          <div className="kpi-tile">
            <div className="kpi-label">Total Final Value</div>
            <div className="kpi-value">${metrics.totalFinal.toLocaleString()}</div>
          </div>
          <div className="kpi-tile">
            <div className="kpi-label">Total PnL</div>
            <div className={`kpi-value ${metrics.totalPnL >= 0 ? 'positive' : 'negative'}`}>
              {metrics.totalPnL >= 0 ? '+' : ''}${metrics.totalPnL.toFixed(2)}
            </div>
          </div>
          <div className="kpi-tile">
            <div className="kpi-label">Average ROI</div>
            <div className={`kpi-value ${metrics.avgROI >= 0 ? 'positive' : 'negative'}`}>
              {metrics.avgROI >= 0 ? '+' : ''}{metrics.avgROI.toFixed(2)}%
            </div>
          </div>
        </div>
      )}

      {/* Leaderboard Table */}
      <div className="leaderboard-section">
        <h3>📊 Full Leaderboard</h3>
        <div className="leaderboard-table">
          <div className="table-header">
            <div className="col-rank">Rank</div>
            <div className="col-agent">Agent</div>
            <div className="col-initial">Initial</div>
            <div className="col-final">Final</div>
            <div className="col-pnl">PnL</div>
            <div className="col-roi">ROI</div>
          </div>
          {leaderboard.map((agent, index) => {
            const initial = agent.initial_value || 10000;
            const final = agent.current_value || agent.final_value || 0;
            const pnl = final - initial;
            const roi = agent.roi || 0;

            return (
              <div key={agent.name} className={`table-row rank-${index + 1}`}>
                <div className="col-rank">
                  {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `${index + 1}`}
                </div>
                <div className="col-agent">{agent.name}</div>
                <div className="col-initial">${initial.toLocaleString()}</div>
                <div className="col-final">${final.toLocaleString()}</div>
                <div className={`col-pnl ${pnl >= 0 ? 'positive' : 'negative'}`}>
                  {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                </div>
                <div className={`col-roi ${roi >= 0 ? 'positive' : 'negative'}`}>
                  {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Back Button */}
      <div className="results-footer">
        <button onClick={onBack} className="back-button-results">
          ← Back to Dashboard
        </button>
      </div>
    </div>
  );
};

export default ResultsDashboard;
