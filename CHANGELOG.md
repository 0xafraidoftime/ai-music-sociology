## Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.
[Unreleased]
Added

Initial project structure and repository setup
Core analysis modules for emotion mapping, earworm studies, and bias detection
Spotify API integration for playlist analysis
Web dashboard using Streamlit
Comprehensive test suite
Documentation for methodology and ethical considerations

Features

Emotional Affordances Mapping: NLP analysis of playlists and lyrics
Algorithmic Earworm Studies: Investigation of recommendation system effects
AI Music Bias Detection: Analysis of cultural biases in generation systems
Interactive Visualizations: Plotly-based dashboards and charts
Data Collection Tools: Spotify and Genius API integrations

Documentation

Comprehensive README with setup instructions
Methodology documentation
Ethical considerations framework
Contributing guidelines
API documentation

Testing

Unit tests for all core modules
Integration tests for API interactions
Mock data for testing without API dependencies

[0.1.0] - 2024-08-22
Added

Initial project scaffolding
Basic repository structure
Core Python modules and requirements
Sample Jupyter notebooks for data exploration
Web application framework
Documentation templates

Infrastructure

GitHub repository setup
CI/CD pipeline configuration
Virtual environment management
Dependency management with requirements.txt


Quick Start Guide
Installation

Clone the repository:

bashgit clone https://github.com/0xafraidoftime/ai-music-sociology.git
cd ai-music-sociology

Set up virtual environment:

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashpip install -r requirements.txt
pip install -e .

Set up environment variables:

bashcp .env.example .env
# Edit .env file with your API keys
Quick Demo
1. Run the Web Dashboard
bashcd web_app
streamlit run app.py
2. Try the Jupyter Notebooks
bashjupyter notebook notebooks/01_data_exploration.ipynb
3. Run Sample Analysis
pythonfrom src.analysis.emotion_mapping import EmotionMapper
from src.analysis.earworm_analysis import EarwormStudy

# Emotion mapping demo
mapper = EmotionMapper()
## ... (see notebooks for full examples)

# Earworm study demo  
study = EarwormStudy()
## ... (see notebooks for full examples)
API Setup
Spotify API

Go to https://developer.spotify.com/dashboard
Create a new app
Copy Client ID and Client Secret to .env file

Genius API

Go to https://genius.com/api-clients
Create a new API client
Copy API key to .env file

Next Steps

Explore the sample notebooks in notebooks/
Run the web dashboard to visualize sample data
Set up real API keys for full functionality
Read the methodology documentation in docs/
Check out the contributing guidelines to get involved

Getting Help

Check the Issues page
Read the full documentation in docs/
Contact the maintainers for research collaboration
