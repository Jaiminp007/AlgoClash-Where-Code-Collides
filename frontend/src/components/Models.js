import React from 'react';
import './Models.css';
import openaiIcon from '../assets/openai.png';
import googleIcon from '../assets/google.png';
import anthropicIcon from '../assets/anthropic.png';
import metaIcon from '../assets/meta.png';
import qwenIcon from '../assets/qwen.png';
import mistralIcon from '../assets/mistral.png';
import deepseekIcon from '../assets/deepseek.png';
import grokIcon from '../assets/grok.png';
import openrouterIcon from '../assets/openrouter.png';
import ModelDirectory from './ModelDirectory';

const modelAssets = [
	{ label: 'OpenAI', img: openaiIcon },
	{ label: 'DeepSeek', img: deepseekIcon },
	{ label: 'Google', img: googleIcon },
	{ label: 'Meta', img: metaIcon },
	{ label: 'Mistral', img: mistralIcon },
	{ label: 'Anthropic', img: anthropicIcon },
	{ label: 'Qwen', img: qwenIcon },
	{ label: 'Grok', img: grokIcon },
	{ label: 'OpenRouter', img: openrouterIcon },
];

const Models = () => {
	// Navigation handler for all nav links
	const handleNav = (e, to) => {
		e.preventDefault();
		if (to === 'home') {
			window.location.href = '/';
		} else if (to === 'models') {
			window.location.href = '/models';
		} else if (to === 'about') {
			window.location.href = '/about';
		} else if (to === 'contact') {
			window.location.href = '/contact';
		}
	};

	// Duplicate the array for seamless infinite scroll effect
	const carouselAssets = [...modelAssets, ...modelAssets];

	return (
		<div className="models-page" style={{ minHeight: '100vh', background: '#000814', color: '#fff' }}>
			<header className="dashboard-header">
				<h1 className="dashboard-title">
					Model Providers
				</h1>
				<nav className="dashboard-nav">
					<a href="#home" onClick={e => handleNav(e, 'home')}>Home</a>
					<a href="/models" className="active" onClick={e => handleNav(e, 'models')}>Models</a>
					<a href="/about" onClick={e => handleNav(e, 'about')}>About</a>
					<a href="/contact" onClick={e => handleNav(e, 'contact')}>Contact</a>
				</nav>
			</header>
			<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', marginTop: 40 }}>
				<div className="models-carousel-outer">
					<div className="models-carousel-track">
						{carouselAssets.map((asset, idx) => (
							<div className="models-carousel-item" key={idx}>
								<img src={asset.img} alt={asset.label + ' icon'} className="models-carousel-img" />
								<div className="models-carousel-label">{asset.label}</div>
							</div>
						))}
					</div>
				</div>
				<div className="models-section">
					<h2 style={{ color: '#FFD60A', marginTop: 30 }}>All Available Models</h2>
					<p style={{ color: '#e6f0ff', marginTop: 6 }}>Sourced live from your backend ai_agents.json, grouped by provider.</p>
					<ModelDirectory compact={false} />
				</div>
			</div>
		</div>
	);
};

export default Models;
