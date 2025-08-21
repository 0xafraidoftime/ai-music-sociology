import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
#Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(file), '..', 'src'))
from analysis.emotion_mapping import EmotionMapper
class TestEmotionMapper:
  @pytest.fixture
def sample_data(self):
    """Create sample data for testing"""
    np.random.seed(42)
    n_tracks = 20
    
    data = {
        'id': [f'track_{i}' for i in range(n_tracks)],
        'name': [f'Song {i}' for i in range(n_tracks)],
        'artist': [f'Artist {i//4}' for i in range(n_tracks)],
        'valence': np.random.beta(2, 2, n_tracks),
        'energy': np.random.beta(2, 2, n_tracks),
        'danceability': np.random.beta(2, 2, n_tracks),
        'acousticness': np.random.beta(2, 5, n_tracks),
        'instrumentalness': np.random.beta(1, 10, n_tracks),
        'liveness': np.random.beta(1, 10, n_tracks),
        'speechiness': np.random.beta(1, 10, n_tracks),
        'popularity': np.random.randint(0, 100, n_tracks),
        'explicit': np.random.choice([True, False], n_tracks),
    }
    
    # Ensure values are in proper ranges
    for feature in ['valence', 'energy', 'danceability', 'acousticness', 
                   'instrumentalness', 'liveness', 'speechiness']:
        data[feature] = np.clip(data[feature], 0, 1)
    
    return pd.DataFrame(data)

@pytest.fixture
def emotion_mapper(self):
    """Create EmotionMapper instance for testing"""
    with patch('analysis.emotion_mapping.EmotionMapper._init_spotify'), \
         patch('analysis.emotion_mapping.EmotionMapper._init_genius'):
        mapper = EmotionMapper()
        mapper.spotify = Mock()
        mapper.genius = Mock()
        return mapper

def test_default_sentiment(self, emotion_mapper):
    """Test default sentiment scores"""
    default_sentiment = emotion_mapper._get_default_sentiment()
    
    assert isinstance(default_sentiment, dict)
    assert 'vader_compound' in default_sentiment
    assert 'textblob_polarity' in default_sentiment
    assert -1 <= default_sentiment['vader_compound'] <= 1
    assert -1 <= default_sentiment['textblob_polarity'] <= 1

def test_create_emotion_clusters(self, emotion_mapper, sample_data):
    """Test emotion clustering functionality"""
    # Add required sentiment columns
    sample_data['vader_compound'] = np.random.uniform(-1, 1, len(sample_data))
    sample_data['textblob_polarity'] = np.random.uniform(-1, 1, len(sample_data))
    
    clustered_data = emotion_mapper.create_emotion_clusters(sample_data, n_clusters=3)
    
    assert 'emotion_cluster' in clustered_data.columns
    assert 'cluster_description' in clustered_data.columns
    assert clustered_data['emotion_cluster'].nunique() <= 3
    assert len(clustered_data) == len(sample_data)

def test_cluster_descriptions(self, emotion_mapper, sample_data):
    """Test that cluster descriptions are generated properly"""
    # Add required sentiment columns
    sample_data['vader_compound'] = np.random.uniform(-1, 1, len(sample_data))
    sample_data['textblob_polarity'] = np.random.uniform(-1, 1, len(sample_data))
    
    clustered_data = emotion_mapper.create_emotion_clusters(sample_data, n_clusters=3)
    
    descriptions = clustered_data['cluster_description'].unique()
    assert len(descriptions) <= 3
    assert all(isinstance(desc, str) for desc in descriptions)

def test_affordance_map_creation(self, emotion_mapper, sample_data):
    """Test creation of affordance map visualization"""
    # Add required sentiment columns
    sample_data['vader_compound'] = np.random.uniform(-1, 1, len(sample_data))
    sample_data['textblob_polarity'] = np.random.uniform(-1, 1, len(sample_data))
    
    clustered_data = emotion_mapper.create_emotion_clusters(sample_data)
    
    # Test PCA method
    fig_pca = emotion_mapper.create_affordance_map(clustered_data, method='pca')
    assert fig_pca is not None
    assert hasattr(fig_pca, 'data')
    
    # Test t-SNE method
    fig_tsne = emotion_mapper.create_affordance_map(clustered_data, method='tsne')
    assert fig_tsne is not None
    assert hasattr(fig_tsne, 'data')
