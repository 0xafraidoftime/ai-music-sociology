import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class EmotionMapper:
    """
    Analyzes playlists and lyrics to map emotional affordances of music
    using NLP and audio feature analysis.
    """
    
    def __init__(self):
        self.spotify = self._init_spotify()
        self.genius = self._init_genius()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.emotion_classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        self.scaler = StandardScaler()
        
    def _init_spotify(self):
        """Initialize Spotify API client"""
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=os.getenv('SPOTIFY_CLIENT_ID'),
                client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
            )
            return spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        except Exception as e:
            print(f"Warning: Spotify API not configured: {e}")
            return None
    
    def _init_genius(self):
        """Initialize Genius API client"""
        try:
            return lyricsgenius.Genius(os.getenv('GENIUS_API_KEY'))
        except Exception as e:
            print(f"Warning: Genius API not configured: {e}")
            return None
    
    def analyze_playlist(self, playlist_id: str) -> pd.DataFrame:
        """
        Comprehensive analysis of a Spotify playlist including audio features
        and lyrical sentiment analysis.
        """
        if not self.spotify:
            raise ValueError("Spotify API not configured")
        
        # Get playlist tracks
        results = self.spotify.playlist_tracks(playlist_id)
        tracks_data = []
        
        for item in results['items']:
            if item['track'] and item['track']['id']:
                track_info = {
                    'id': item['track']['id'],
                    'name': item['track']['name'],
                    'artist': item['track']['artists'][0]['name'],
                    'popularity': item['track']['popularity'],
                    'explicit': item['track']['explicit']
                }
                tracks_data.append(track_info)
        
        df = pd.DataFrame(tracks_data)
        
        # Get audio features
        track_ids = df['id'].tolist()
        audio_features = self.spotify.audio_features(track_ids)
        
        audio_df = pd.DataFrame([f for f in audio_features if f])
        df = df.merge(audio_df, on='id', how='left')
        
        # Analyze lyrics sentiment
        df['lyrics_sentiment'] = df.apply(self._analyze_track_sentiment, axis=1)
        
        # Extract emotion scores
        emotion_scores = df['lyrics_sentiment'].apply(pd.Series)
        df = pd.concat([df, emotion_scores], axis=1)
        
        return df
    
    def _analyze_track_sentiment(self, track_row) -> Dict:
        """Analyze sentiment and emotions for a single track"""
        if not self.genius:
            return self._get_default_sentiment()
        
        try:
            # Search for lyrics
            song = self.genius.search_song(track_row['name'], track_row['artist'])
            if not song or not song.lyrics:
                return self._get_default_sentiment()
            
            lyrics = song.lyrics
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(lyrics)
            
            # TextBlob sentiment
            blob = TextBlob(lyrics)
            textblob_sentiment = blob.sentiment
            
            # Emotion classification
            emotions = self.emotion_classifier(lyrics[:512])  # Limit text length
            emotion_scores = {e['label']: e['score'] for e in emotions[0]}
            
            return {
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'textblob_polarity': textblob_sentiment.polarity,
                'textblob_subjectivity': textblob_sentiment.subjectivity,
                **emotion_scores
            }
            
        except Exception as e:
            print(f"Error analyzing lyrics for {track_row['name']}: {e}")
            return self._get_default_sentiment()
    
    def _get_default_sentiment(self) -> Dict:
        """Return default sentiment scores when lyrics analysis fails"""
        return {
            'vader_compound': 0.0,
            'vader_positive': 0.33,
            'vader_negative': 0.33,
            'vader_neutral': 0.33,
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.5,
            'joy': 0.2, 'sadness': 0.2, 'anger': 0.2, 'fear': 0.2, 'surprise': 0.2
        }
    
    def create_emotion_clusters(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Create emotion-based clusters of songs"""
        # Select features for clustering
        feature_cols = [
            'valence', 'energy', 'danceability', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness',
            'vader_compound', 'textblob_polarity'
        ]
        
        # Handle missing values
        cluster_data = df[feature_cols].fillna(df[feature_cols].mean())
        
        # Scale features
        scaled_features = self.scaler.fit_transform(cluster_data)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['emotion_cluster'] = kmeans.fit_predict(scaled_features)
        
        # Create cluster descriptions
        cluster_descriptions = self._describe_clusters(df, feature_cols)
        df['cluster_description'] = df['emotion_cluster'].map(cluster_descriptions)
        
        return df
    
    def _describe_clusters(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Generate descriptive names for emotion clusters"""
        descriptions = {}
        
        for cluster in df['emotion_cluster'].unique():
            cluster_data = df[df['emotion_cluster'] == cluster][feature_cols]
            means = cluster_data.mean()
            
            # Simple heuristic for naming clusters
            if means['valence'] > 0.6 and means['energy'] > 0.6:
                descriptions[cluster] = "Energetic & Positive"
            elif means['valence'] < 0.4 and means['energy'] < 0.4:
                descriptions[cluster] = "Melancholic & Calm"
            elif means['energy'] > 0.7:
                descriptions[cluster] = "High Energy"
            elif means['acousticness'] > 0.6:
                descriptions[cluster] = "Acoustic & Intimate"
            else:
                descriptions[cluster] = f"Cluster {cluster}"
        
        return descriptions
    
    def create_affordance_map(self, df: pd.DataFrame, method: str = 'pca') -> go.Figure:
        """Create 2D visualization of psychological music spaces"""
        feature_cols = [
            'valence', 'energy', 'danceability', 'acousticness',
            'vader_compound', 'textblob_polarity'
        ]
        
        cluster_data = df[feature_cols].fillna(df[feature_cols].mean())
        scaled_features = self.scaler.fit_transform(cluster_data)
        
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(scaled_features)
            x_label = f"PC1 ({reducer.explained_variance_ratio_[0]:.2f})"
            y_label = f"PC2 ({reducer.explained_variance_ratio_[1]:.2f})"
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42)
            coords = reducer.fit_transform(scaled_features)
            x_label = "t-SNE 1"
            y_label = "t-SNE 2"
        
        df_plot = df.copy()
        df_plot['x'] = coords[:, 0]
        df_plot['y'] = coords[:, 1]
        
        fig = px.scatter(
            df_plot, x='x', y='y',
            color='cluster_description',
            hover_data=['name', 'artist', 'valence', 'energy'],
            title="Emotional Affordances Map of Music",
            labels={'x': x_label, 'y': y_label}
        )
        
        return fig
