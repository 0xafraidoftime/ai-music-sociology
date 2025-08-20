import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.emotion_mapping import EmotionMapper
from analysis.earworm_analysis import EarwormStudy
from analysis.bias_detection import BiasAnalyzer
from data_collection.spotify_scraper import SpotifyPlaylistAnalyzer

st.set_page_config(
    page_title="AI & Music Sociology Dashboard",
    page_icon="🎵",
    layout="wide"
)

def main():
    st.title("🎵 AI & Music Sociology Research Dashboard")
    st.markdown("Exploring the intersection of artificial intelligence, music, and social behavior")
    
    # Sidebar for navigation
    st.sidebar.title("Research Modules")
    module = st.sidebar.selectbox(
        "Choose analysis module:",
        ["Emotional Affordances", "Earworm Studies", "AI Bias Detection", "Playlist Analysis"]
    )
    
    if module == "Emotional Affordances":
        emotional_affordances_page()
    elif module == "Earworm Studies":
        earworm_studies_page()
    elif module == "AI Bias Detection":
        bias_detection_page()
    elif module == "Playlist Analysis":
        playlist_analysis_page()

def emotional_affordances_page():
    st.header("🎭 Emotional Affordances Mapping")
    st.markdown("Analyze how music creates psychological spaces through audio features and lyrical content")
    
    # Initialize emotion mapper
    emotion_mapper = EmotionMapper()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # Sample data option
        use_sample_data = st.checkbox("Use sample data", value=True)
        
        if not use_sample_data:
            playlist_url = st.text_input("Spotify Playlist URL:")
            if playlist_url and st.button("Analyze Playlist"):
                playlist_id = playlist_url.split('/')[-1].split('?')[0]
                with st.spinner("Analyzing playlist..."):
                    try:
                        df = emotion_mapper.analyze_playlist(playlist_id)
                        st.session_state['emotion_data'] = df
                        st.success(f"Analyzed {len(df)} tracks!")
                    except Exception as e:
                        st.error(f"Error analyzing playlist: {e}")
        else:
            # Generate sample data
            if st.button("Generate Sample Analysis"):
                df = generate_sample_emotion_data()
                st.session_state['emotion_data'] = df
                st.success("Generated sample emotional analysis data!")
    
    with col2:
        if 'emotion_data' in st.session_state:
            df = st.session_state['emotion_data']
            
            # Create clusters
            df_clustered = emotion_mapper.create_emotion_clusters(df)
            
            # Display cluster visualization
            fig = emotion_mapper.create_affordance_map(df_clustered)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cluster statistics
            st.subheader("Emotional Clusters")
            cluster_stats = df_clustered.groupby('cluster_description').agg({
                'valence': 'mean',
                'energy': 'mean',
                'danceability': 'mean',
                'vader_compound': 'mean'
            }).round(3)
            st.dataframe(cluster_stats)

def earworm_studies_page():
    st.header("🔄 Algorithmic Earworm Studies")
    st.markdown("Investigate how AI recommendation systems influence song persistence and mood")
    
    study = EarwormStudy()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Study Parameters")
        
        n_participants = st.slider("Participants per condition:", 10, 100, 30)
        
        sample_tracks = [
            {'id': 'track1', 'name': 'Blinding Lights', 'artist': 'The Weeknd'},
            {'id': 'track2', 'name': 'Shape of You', 'artist': 'Ed Sheeran'},
            {'id': 'track3', 'name': 'Don\'t Stop Me Now', 'artist': 'Queen'}
        ]
        
        if st.button("Run Earworm Study"):
            with st.spinner("Running comparative study..."):
                df = study.run_comparative_study(sample_tracks, n_participants)
                results = study.analyze_persistence_patterns(df)
                
                st.session_state['earworm_data'] = df
                st.session_state['earworm_results'] = results
                st.success("Study completed!")
    
    with col2:
        if 'earworm_data' in st.session_state and 'earworm_results' in st.session_state:
            df = st.session_state['earworm_data']
            results = st.session_state['earworm_results']
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Earworm Persistence', 'Mood Change', 
                               'Repetition vs Persistence', 'Distribution'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Box plots for persistence and mood change
            for i, condition in enumerate(df['repetition_type'].unique()):
                subset = df[df['repetition_type'] == condition]
                fig.add_trace(
                    go.Box(y=subset['earworm_persistence'], name=condition, showlegend=False),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Box(y=subset['mood_change'], name=condition, showlegend=False),
                    row=1, col=2
                )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df['repetition_count'], 
                    y=df['earworm_persistence'],
                    mode='markers',
                    marker=dict(color=df['repetition_type'].astype('category').cat.codes),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Histogram
            for condition in df['repetition_type'].unique():
                subset = df[df['repetition_type'] == condition]
                fig.add_trace(
                    go.Histogram(x=subset['earworm_persistence'], name=condition, opacity=0.7),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, title_text="Earworm Study Results")
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical results
            st.subheader("Statistical Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                st.metric("ANOVA F-statistic", f"{results['anova']['f_statistic']:.3f}")
                st.metric("P-value", f"{results['anova']['p_value']:.6f}")
                
            with col4:
                # Condition means
                means_df = df.groupby('repetition_type')['earworm_persistence'].mean().round(2)
                st.write("**Mean Persistence (hours):**")
                for condition, mean_val in means_df.items():
                    st.write(f"- {condition}: {mean_val}")

def bias_detection_page():
    st.header("⚖️ AI Music Generation Bias Detection")
    st.markdown("Examine cultural and genre biases in AI music generation tools")
    
    analyzer = BiasAnalyzer()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Bias Analysis")
        
        selected_cultures = st.multiselect(
            "Select cultural contexts to analyze:",
            options=list(analyzer.cultural_prompts.keys()),
            default=['western_contemporary', 'african', 'asian', 'latin']
        )
        
        if st.button("Run Bias Analysis"):
            with st.spinner("Generating and analyzing music samples..."):
                # Filter prompts based on selection
                filtered_prompts = {k: v for k, v in analyzer.cultural_prompts.items() 
                                  if k in selected_cultures}
                analyzer.cultural_prompts = filtered_prompts
                
                df = analyzer.run_comprehensive_bias_test()
                metrics = analyzer.calculate_bias_metrics(df)
                
                st.session_state['bias_data'] = df
                st.session_state['bias_metrics'] = metrics
                st.success("Bias analysis completed!")
    
    with col2:
        if 'bias_data' in st.session_state and 'bias_metrics' in st.session_state:
            df = st.session_state['bias_data']
            metrics = st.session_state['bias_metrics']
            
            # Interactive PCA visualization
            fig = analyzer.create_interactive_dashboard(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Bias severity scores
            st.subheader("Bias Severity Rankings")
            if 'bias_severity_scores' in metrics:
                bias_df = pd.DataFrame(
                    list(metrics['bias_severity_scores'].items()),
                    columns=['Cultural Context', 'Bias Score']
                ).sort_values('Bias Score', ascending=False)
                
                fig_bias = px.bar(
                    bias_df, 
                    x='Cultural Context', 
                    y='Bias Score',
                    title='Bias Severity by Cultural Context'
                )
                st.plotly_chart(fig_bias, use_container_width=True)
                
                # Show detailed metrics
                with st.expander("Detailed Statistical Results"):
                    st.write("**Western vs Non-Western Comparisons:**")
                    for feature, result in metrics['western_vs_nonwestern_tests'].items():
                        significance = "***" if result['p_value'] < 0.001 else \
                                     "**" if result['p_value'] < 0.01 else \
                                     "*" if result['p_value'] < 0.05 else ""
                        st.write(f"- {feature}: p={result['p_value']:.6f} {significance}")

def playlist_analysis_page():
    st.header("📊 Spotify Playlist Analysis")
    st.markdown("Analyze Spotify playlists for musical and emotional characteristics")
    
    analyzer = SpotifyPlaylistAnalyzer()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Playlist Input")
        
        # Option to use sample data or real Spotify data
        data_source = st.radio(
            "Data source:",
            ["Sample Data", "Spotify API"]
        )
        
        if data_source == "Spotify API":
            playlist_url = st.text_input(
                "Spotify Playlist URL:",
                placeholder="https://open.spotify.com/playlist/..."
            )
            
            if playlist_url and st.button("Analyze Playlist"):
                playlist_id = extract_playlist_id(playlist_url)
                if playlist_id:
                    with st.spinner("Fetching playlist data..."):
                        try:
                            df = analyzer.get_complete_playlist_analysis(playlist_id)
                            st.session_state['playlist_data'] = df
                            st.success(f"Analyzed {len(df)} tracks!")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.info("Try using sample data instead.")
                else:
                    st.error("Invalid playlist URL")
        else:
            if st.button("Load Sample Playlist"):
                df = generate_sample_playlist_data()
                st.session_state['playlist_data'] = df
                st.success("Loaded sample playlist data!")
    
    with col2:
        if 'playlist_data' in st.session_state:
            df = st.session_state['playlist_data']
            
            # Basic statistics
            st.subheader("Playlist Overview")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Total Tracks", len(df))
            with col4:
                avg_popularity = df['popularity'].mean() if 'popularity' in df.columns else 0
                st.metric("Avg Popularity", f"{avg_popularity:.1f}")
            with col5:
                avg_duration = df['duration_ms'].mean() / 60000 if 'duration_ms' in df.columns else 0
                st.metric("Avg Duration", f"{avg_duration:.1f} min")
            
            # Audio features radar chart
            if all(col in df.columns for col in ['valence', 'energy', 'danceability', 'acousticness']):
                st.subheader("Audio Features Profile")
                
                features = ['valence', 'energy', 'danceability', 'acousticness', 
                           'instrumentalness', 'liveness', 'speechiness']
                feature_means = [df[f].mean() for f in features if f in df.columns]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=feature_means,
                    theta=features[:len(feature_means)],
                    fill='toself',
                    name='Playlist Profile'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True,
                    title="Playlist Audio Features Profile"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Top artists and tracks
            st.subheader("Top Artists")
            if 'artist_name' in df.columns:
                top_artists = df['artist_name'].value_counts().head(10)
                fig_artists = px.bar(
                    x=top_artists.values, 
                    y=top_artists.index,
                    orientation='h',
                    title="Most Frequent Artists"
                )
                fig_artists.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_artists, use_container_width=True)
            
            # Show raw data
            with st.expander("View Raw Data"):
                st.dataframe(df)

def generate_sample_emotion_data():
    """Generate sample data for emotion mapping demonstration"""
    np.random.seed(42)
    
    n_tracks = 50
    data = {
        'track_id': [f"track_{i}" for i in range(n_tracks)],
        'name': [f"Song {i}" for i in range(n_tracks)],
        'artist': [f"Artist {i//5}" for i in range(n_tracks)],
        'valence': np.random.beta(2, 2, n_tracks),
        'energy': np.random.beta(2, 2, n_tracks),
        'danceability': np.random.beta(2, 2, n_tracks),
        'acousticness': np.random.beta(2, 5, n_tracks),
        'instrumentalness': np.random.beta(1, 10, n_tracks),
        'liveness': np.random.beta(1, 10, n_tracks),
        'speechiness': np.random.beta(1, 10, n_tracks),
        'vader_compound': np.random.normal(0, 0.5, n_tracks),
        'textblob_polarity': np.random.normal(0, 0.3, n_tracks),
    }
    
    # Clip values to reasonable ranges
    for key in ['valence', 'energy', 'danceability', 'acousticness', 
                'instrumentalness', 'liveness', 'speechiness']:
        data[key] = np.clip(data[key], 0, 1)
    
    data['vader_compound'] = np.clip(data['vader_compound'], -1, 1)
    data['textblob_polarity'] = np.clip(data['textblob_polarity'], -1, 1)
    
    return pd.DataFrame(data)

def generate_sample_playlist_data():
    """Generate sample playlist data"""
    np.random.seed(42)
    
    n_tracks = 30
    artists = ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E']
    
    data = {
        'track_id': [f"track_{i}" for i in range(n_tracks)],
        'track_name': [f"Song Title {i}" for i in range(n_tracks)],
        'artist_name': np.random.choice(artists, n_tracks),
        'album_name': [f"Album {i//3}" for i in range(n_tracks)],
        'popularity': np.random.randint(20, 100, n_tracks),
        'duration_ms': np.random.normal(210000, 60000, n_tracks),  # ~3.5 min average
        'valence': np.random.beta(2, 2, n_tracks),
        'energy': np.random.beta(2, 2, n_tracks),
        'danceability': np.random.beta(2, 2, n_tracks),
        'acousticness': np.random.beta(2, 5, n_tracks),
        'instrumentalness': np.random.beta(1, 10, n_tracks),
        'liveness': np.random.beta(1, 10, n_tracks),
        'speechiness': np.random.beta(1, 10, n_tracks),
        'tempo': np.random.normal(120, 30, n_tracks),
    }
    
    # Ensure reasonable ranges
    data['duration_ms'] = np.clip(data['duration_ms'], 60000, 600000)  # 1-10 minutes
    data['tempo'] = np.clip(data['tempo'], 60, 200)
    
    for key in ['valence', 'energy', 'danceability', 'acousticness', 
                'instrumentalness', 'liveness', 'speechiness']:
        data[key] = np.clip(data[key], 0, 1)
    
    return pd.DataFrame(data)

def extract_playlist_id(url: str) -> str:
    """Extract playlist ID from Spotify URL"""
    try:
        if 'playlist/' in url:
            return url.split('playlist/')[1].split('?')[0]
        return None
    except:
        return None

if __name__ == "__main__":
    main()

---

# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-music-sociology",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Research tools for investigating AI's impact on music and social behavior",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-music-sociology",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Sociology",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "nltk>=3.7",
        "textblob>=0.17.1",
        "vaderSentiment>=3.3.2",
        "transformers>=4.20.0",
        "spacy>=3.4.0",
        "librosa>=0.9.2",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "streamlit>=1.12.0",
        "spotipy>=2.20.0",
        "requests>=2.28.0",
        "python-dotenv>=0.20.0",
        "flask>=2.2.0",
        "lyricsgenius>=3.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ai-music-sociology=src.cli:main",
        ],
    },
)
