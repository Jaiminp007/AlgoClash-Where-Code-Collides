# Frontend - AlgoClash

React-based web interface for AlgoClash, providing an interactive dashboard for AI trading algorithm generation, simulation, and results visualization.

## Overview

The frontend is a modern React application that provides a seamless user experience for:
- Selecting AI models and stock symbols
- Generating trading algorithms in real-time
- Running market simulations with live updates
- Visualizing performance through interactive charts
- Comparing algorithm results

## Structure

```
frontend/
├── public/                 # Static assets
│   ├── index.html         # HTML template
│   └── favicon.ico        # App icon
├── src/
│   ├── components/        # React components
│   │   ├── Dashboard.js   # Main application container
│   │   ├── Dashboard.css  # Dashboard styling
│   │   ├── MarketSimulationChart.js  # Real-time chart component
│   │   └── MarketSimulationChart.css # Chart styling
│   ├── App.js             # Root component
│   ├── App.css            # Global styles
│   └── index.js           # Application entry point
└── package.json           # Dependencies and scripts
```

## Key Components

### 1. Dashboard (`Dashboard.js`)
The main application container that manages:
- **AI Model Selection**: Grid of 60+ AI models from various providers
- **Stock Symbol Selection**: Dropdown for available stock tickers
- **Generation Phase Management**: State machine for workflow control
- **Real-time Progress Tracking**: Polling-based updates during generation/simulation
- **Results Display**: Final rankings and performance metrics

**State Machine:**
- `idle` → Initial state, ready to start
- `generating` → AI algorithms being generated
- `review` → User reviews generated algorithms
- `simulating` → Market simulation running
- `completed` → Results ready for viewing

### 2. Market Simulation Chart (`MarketSimulationChart.js`)
Interactive Recharts-based visualization displaying:
- **Dual Y-Axis Chart**: Market price (left) + Portfolio values (right)
- **Real-time Updates**: Live data streaming during simulation
- **Agent Performance Lines**: Color-coded portfolio trajectories
- **Custom Tooltip**: Detailed agent info on hover
- **PDF Export**: Download chart as high-quality PDF
- **Agent Statistics Cards**: Side panel with ROI, cash, and stock holdings

**Chart Features:**
- Responsive container adapts to screen size
- Auto-scaling axes for optimal visibility
- Grid and labels for easy reading
- Legend with color mapping
- Smooth animations

### 3. Agent Selection Grid
- **60+ AI Models**: Claude, GPT, Gemini, DeepSeek, Llama, and more
- **Provider Grouping**: Visual organization by AI provider
- **Selection Limits**: Minimum 2, maximum 10 agents
- **Visual Feedback**: Selected state with checkmarks
- **Search/Filter**: (Future enhancement)

### 4. Algorithm Preview Modal
- **Code Display**: Syntax-highlighted Python code
- **Model Information**: Shows which AI generated the algorithm
- **Full-screen View**: Modal overlay for comfortable reading
- **Replace Functionality**: Option to regenerate with different model

### 5. Results Overlay
- **Final Rankings**: Leaderboard with podium (🥇🥈🥉)
- **Performance Metrics**: ROI, trades, final positions
- **Full Chart View**: Complete 0-250 tick visualization
- **Close Button**: Return to dashboard for new simulation

## Data Flow

```
User Action
    ↓
Dashboard State Update
    ↓
API Call (POST /api/generate)
    ↓
Polling Loop (GET /api/generation/:id)
    ↓
Real-time Progress Updates
    ↓
Algorithm Review
    ↓
Simulation Start (POST /api/simulate/:id)
    ↓
Polling Loop (GET /api/simulation/:id)
    ↓
Chart Data Updates
    ↓
Results Display
```

## Key Features

### Real-time Generation
- **Per-Model Progress**: Individual status for each AI (generating/done/error)
- **Live Code Previews**: See algorithms as they're generated
- **Failure Handling**: Replace failed models without restarting
- **Progress Bar**: Visual indication of overall completion

### Interactive Simulation
- **Live Chart Updates**: Portfolio values update every tick
- **Trade Indicators**: Visual markers for buy/sell actions
- **Performance Cards**: Real-time ROI calculations
- **Status Messages**: Informative feedback throughout

### Results Visualization
- **Podium Display**: Top 3 agents prominently featured
- **Detailed Stats**: Comprehensive metrics for all agents
- **Chart Export**: Download results as PDF
- **Full History**: Complete tick-by-tick data

## Styling

The UI features a dark, professional theme:
- **Primary Color**: `#6366f1` (Indigo)
- **Background**: `#1a1a2e` (Dark Navy)
- **Cards**: `#16213e` with subtle borders
- **Text**: White with high contrast
- **Accents**: Gradient overlays and hover effects

**Design Principles:**
- Clean, modern interface
- High readability
- Responsive layout
- Smooth transitions
- Intuitive navigation

## API Integration

### Environment Configuration
```javascript
// Uses environment variable or defaults to same host
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';
```

### API Endpoints Used

**1. Get AI Agents**
```javascript
GET /api/ai_agents
Response: { agents: { anthropic: [...], openai: [...], ... } }
```

**2. Get Stock Symbols**
```javascript
GET /api/data_files
Response: { stocks: [{ ticker: "AAPL", filename: "AAPL_data.csv" }, ...] }
```

**3. Start Generation**
```javascript
POST /api/generate
Body: { agents: ["model1", "model2"], stock: "AAPL_data.csv" }
Response: { generation_id: "gen_123...", status: "started" }
```

**4. Poll Generation Status**
```javascript
GET /api/generation/:id
Response: {
  status: "running",
  progress: 45,
  algorithms: { "model1": "code...", ... },
  model_states: { "model1": "done", "model2": "generating" }
}
```

**5. Start Simulation**
```javascript
POST /api/simulate/:generation_id
Response: { simulation_id: "sim_456...", status: "started" }
```

**6. Poll Simulation Status**
```javascript
GET /api/simulation/:id
Response: {
  status: "running",
  progress: 75,
  chart_data: [{ tick: 0, price: 150.00, agent_portfolios: {...} }, ...],
  results: null
}
```

### Polling Strategy
- **Generation**: Poll every 500ms during algorithm generation
- **Simulation**: Poll every 300ms during market simulation
- **Auto-cleanup**: Stop polling on completion/error
- **Error Recovery**: Graceful degradation with status messages

## Installation & Setup

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## Environment Variables

Create `.env` file:
```bash
REACT_APP_API_BASE_URL=http://localhost:5000
```

## Key Dependencies

- **react** (^18.x): UI framework
- **recharts** (^2.x): Charting library
- **html2canvas** (^1.x): Chart screenshot for PDF
- **jspdf** (^2.x): PDF generation

## Component Communication

```
Dashboard (Parent)
├── Header/Nav
├── Agent Selection Grid
│   └── Agent Cards (60+)
├── Stock Selector
├── Progress Bar
├── Generation Status
├── MarketSimulationChart
│   ├── Chart Container
│   ├── Tooltip
│   ├── Legend
│   └── Stats Cards
├── Results Overlay
│   ├── Rankings
│   └── Final Chart
├── Preview Modal
│   └── Code Display
└── Replace Modal
    └── New Agent Selector
```

## State Management

Uses React hooks for state:
- `useState` - Local component state
- `useEffect` - Side effects (API calls, polling)
- `useRef` - Mutable refs (polling intervals, chart refs)
- `useMemo` - Computed values (chart data processing)

**Key State Variables:**
```javascript
selectedAgents          // Selected AI models
selectedStock           // Chosen stock symbol
generationPhase         // Workflow state
progress                // 0-100 percentage
chartData               // Real-time simulation data
simulationResults       // Final rankings
generatedAlgos          // Generated algorithm code
currentGenId            // Active generation ID
currentSimId            // Active simulation ID
```

## Performance Optimizations

1. **Polling Intervals**: Configurable timing to balance responsiveness vs. server load
2. **Chart Data Limiting**: Backend limits to 250 ticks to prevent memory issues
3. **Memoization**: Chart data processing cached with `useMemo`
4. **Conditional Rendering**: Components only render when phase matches
5. **Session Storage**: Persist generation ID across refreshes

## Browser Compatibility

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Requirements:**
- Modern ES6+ support
- CSS Grid and Flexbox
- Fetch API
- Local Storage

## Development Tips

### Debug Mode
```javascript
// Add to Dashboard.js for verbose logging
useEffect(() => {
  console.log('State:', {
    generationPhase,
    progress,
    chartDataLength: chartData.length
  });
}, [generationPhase, progress, chartData]);
```

### Mock Data Testing
```javascript
// Test chart without backend
const mockChartData = Array.from({ length: 100 }, (_, i) => ({
  tick: i,
  price: 150 + Math.random() * 10,
  agent_portfolios: {
    Agent1: { value: 10000 + i * 50, cash: 5000, stock: 30 }
  }
}));
setChartData(mockChartData);
```

### Hot Reload
Development server supports hot module replacement - changes appear instantly without full page refresh.

## Common Issues & Solutions

**Chart not rendering:**
- Check `chartData` is non-empty array
- Verify data has required fields: `tick`, `price`, `agent_portfolios`
- Inspect console for Recharts errors

**Polling stuck:**
- Clear intervals in cleanup: `return () => clearInterval(pollingRef.current)`
- Check network tab for failed requests
- Verify backend is running on correct port

**Agent selection not working:**
- Check selected count doesn't exceed max (10)
- Verify agent ID format matches backend expectations
- Review console for selection state changes

**PDF export fails:**
- Ensure html2canvas loaded: `npm install html2canvas`
- Check browser console for CORS issues
- Verify chart is fully rendered before export

## Future Enhancements

- [ ] WebSocket for real-time updates (replace polling)
- [ ] Multiple simulation comparison view
- [ ] Historical simulation replay
- [ ] Advanced filtering/search for AI models
- [ ] Mobile-responsive design improvements
- [ ] Dark/light theme toggle
- [ ] Shareable result URLs
- [ ] Algorithm code editor
- [ ] Custom strategy upload

## Technologies

- **React 18** - Component framework
- **Recharts** - Data visualization
- **CSS3** - Modern styling with Grid/Flexbox
- **Fetch API** - HTTP requests
- **jsPDF + html2canvas** - PDF generation

---

**Ready to generate algorithms? Select your AI models and click Start Generation!** 🚀
