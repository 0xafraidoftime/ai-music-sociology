# AI & Music Sociology Research Repository

**Exploring the Intersection of Artificial Intelligence, Music, and Social Behavior**

## Overview

This repository investigates how AI systems shape musical experiences, preferences, and cultural production through three interconnected research streams:

1. **Emotional Affordances Mapping** - NLP analysis of playlists and lyrics to understand psychological music spaces
2. **Algorithmic Earworms** - Studies on how recommendation systems influence song persistence and mood
3. **Generative AI Bias Analysis** - Examining cultural and genre biases in AI music generation tools

## Repository Structure

```
ai-music-sociology/
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
├── data/
│   ├── raw/
│   │   ├── playlists/
│   │   ├── lyrics/
│   │   └── generated_samples/
│   ├── processed/
│   └── external/
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── spotify_scraper.py
│   │   ├── lyrics_fetcher.py
│   │   └── music_generation_api.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── emotion_mapping.py
│   │   ├── earworm_analysis.py
│   │   ├── bias_detection.py
│   │   └── clustering_utils.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── dashboards.py
│   │   ├── emotion_plots.py
│   │   └── bias_charts.py
│   └── utils/
│       ├── __init__.py
│       ├── nlp_utils.py
│       └── audio_utils.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_emotion_affordances.ipynb
│   ├── 03_earworm_persistence.ipynb
│   └── 04_bias_analysis.ipynb
├── experiments/
│   ├── earworm_study/
│   ├── recommendation_influence/
│   └── cultural_bias_tests/
├── web_app/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── requirements.txt
├── docs/
│   ├── methodology.md
│   ├── ethical_considerations.md
│   └── api_documentation.md
├── tests/
└── results/
    ├── figures/
    ├── reports/
    └── datasets/
```

## Core Research Components

### 1. Emotional Affordances Mapping (`src/analysis/emotion_mapping.py`)

**Objective:** Use NLP to map the emotional and psychological "affordances" of music through playlist and lyric analysis.

**Key Features:**
- Spotify playlist analysis via API
- Lyrics sentiment analysis using VADER, TextBlob, and transformer models
- K-means clustering of songs by emotional dimensions
- Visualization of "psychological music spaces"
- Cross-cultural emotion mapping

**Methodology:**
- Extract audio features (valence, energy, danceability) from Spotify API
- Perform sentiment analysis on lyrics using multiple NLP models
- Apply dimensionality reduction (PCA, t-SNE) for visualization
- Create emotional clusters and affordance maps

### 2. Algorithmic Earworms Study (`src/analysis/earworm_analysis.py`)

**Objective:** Investigate how AI recommendation systems influence earworm persistence and mood regulation.

**Key Features:**
- Simulation of algorithmic repetition effects
- Survey integration for participant feedback
- Repetition pattern analysis
- Mood tracking over listening sessions
- Comparison of AI-curated vs. self-selected repetition

**Experimental Design:**
- Control group: Self-selected song repetition
- Treatment group: AI-recommended song loops (Spotify autoplay simulation)
- Measure: Earworm persistence, mood changes, listening fatigue
- Data collection: Post-listening surveys, behavioral tracking

### 3. Generative AI Bias Detection (`src/analysis/bias_detection.py`)

**Objective:** Examine cultural, genre, and demographic biases in AI music generation tools.

**Key Features:**
- API integration with music generation services (Suno, AIVA, Jukebox)
- Cross-cultural prompt testing
- Musical feature analysis (tempo, key, instrumentation)
- Diversity metrics calculation
- Bias visualization and reporting

**Analysis Framework:**
- Generate music samples across diverse cultural prompts
- Extract musical features using librosa and essentia
- Statistical analysis of feature distributions by culture/genre
- Bias quantification using fairness metrics

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/0xafraidoftime/ai-music-sociology.git
cd ai-music-sociology

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys (Spotify, music generation services)

# Run initial setup
python setup.py install
```

## Required API Keys

```bash
# Add to your .env file
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
GENIUS_API_KEY=your_genius_key
SUNO_API_KEY=your_suno_key  # If available
OPENAI_API_KEY=your_openai_key  # For GPT-based analysis
```

## Quick Start

### 1. Emotional Affordances Analysis

```python
from src.analysis.emotion_mapping import EmotionMapper
from src.data_collection.spotify_scraper import SpotifyPlaylistAnalyzer

# Initialize analyzers
emotion_mapper = EmotionMapper()
playlist_analyzer = SpotifyPlaylistAnalyzer()

# Analyze a playlist
playlist_id = "37i9dQZF1DXcBWIGoYBM5M"  # Example: Today's Top Hits
emotional_profile = emotion_mapper.analyze_playlist(playlist_id)

# Generate emotion map
emotion_mapper.create_affordance_map(emotional_profile)
```

### 2. Earworm Persistence Study

```python
from src.analysis.earworm_analysis import EarwormStudy

study = EarwormStudy()

# Simulate algorithmic repetition
results = study.run_repetition_experiment(
    track_id="4iV5W9uYEdYUVa79Axb7Rh",  # Example track
    repetition_type="algorithmic",
    duration_minutes=30
)

study.analyze_persistence_patterns(results)
```

### 3. AI Music Bias Analysis

```python
from src.analysis.bias_detection import BiasAnalyzer

bias_analyzer = BiasAnalyzer()

# Test cultural bias in generation
cultural_prompts = [
    "traditional indian classical music",
    "african drumming patterns",
    "western pop ballad",
    "latin salsa rhythm"
]

bias_results = bias_analyzer.test_cultural_bias(cultural_prompts)
bias_analyzer.visualize_bias_patterns(bias_results)
```

## Web Dashboard

Launch the interactive dashboard to explore results:

```bash
cd web_app
python app.py
```

Features:
- Real-time playlist emotion analysis
- Earworm persistence visualization
- Bias detection results
- Interactive exploration of musical spaces

## Research Methodologies

### Data Collection Ethics
- All data collection follows platform ToS
- User privacy protection measures
- Anonymized participant data
- IRB compliance considerations

### Statistical Approaches
- Multi-level modeling for nested data structures
- Causal inference methods for recommendation effects
- Fairness metrics for bias quantification
- Cross-validation for model robustness

### Validation Techniques
- Inter-rater reliability for emotion labeling
- Cross-platform validation of findings
- Temporal stability testing
- Cultural sensitivity review

## Key Dependencies

```txt
# Core Data Science
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
scipy>=1.9.0

# NLP & Text Analysis
nltk>=3.7
textblob>=0.17.1
vaderSentiment>=3.3.2
transformers>=4.20.0
spacy>=3.4.0

# Audio Analysis
librosa>=0.9.2
essentia>=2.1b6.dev

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
streamlit>=1.12.0

# API Integrations
spotipy>=2.20.0
requests>=2.28.0
python-dotenv>=0.20.0

# Web Framework
flask>=2.2.0
dash>=2.6.0
```

## Contributing

We welcome contributions from researchers, developers, and musicians! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Research methodology requirements
- Ethical considerations
- Peer review process

## Academic Citations

If you use this repository in your research, please cite:

```bibtex
@misc{ai_music_sociology_2024,
  title={AI \& Music Sociology: Investigating Algorithmic Influence on Musical Experience},
  author={[Ankita Pal]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/0xafraidoftime/ai-music-sociology}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Spotify Web API for music data access
- Genius API for lyrics data
- Open-source music generation models
- Research participants and collaborators

## Future Research Directions

- Cross-platform recommendation system comparison
- Longitudinal studies of music taste evolution
- Neuroscience integration (EEG during listening)
- Social network analysis of music sharing
- AI-human collaborative composition studies

---

**Contact:** [2022AC05327@wilp.bits-pilani.ac.in]
**Project Status:** Active Development
**Last Updated:** August 2024
