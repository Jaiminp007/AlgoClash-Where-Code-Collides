import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Models from './components/Models';
import About from './components/About';
import Contact from './components/Contact';
import ReviewScreen from './components/ReviewScreen';
import SimulationResults from './components/SimulationResults';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/models" element={<Models />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/review" element={<ReviewScreen />} />
          <Route path="/simulation-results" element={<SimulationResults />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
