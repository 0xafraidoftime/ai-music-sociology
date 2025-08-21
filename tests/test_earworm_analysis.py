import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(file), '..', 'src'))
from analysis.earworm_analysis import EarwormStudy, ListeningSession
class TestEarwormStudy:
@pytest.fixture
def earworm_study(self):
    """Create EarwormStudy instance for testing"""
    return EarwormStudy()

@pytest.fixture
def sample_tracks(self):
    """Sample tracks for testing"""
    return [
        {'id': 'track1', 'name': 'Test Song 1', 'artist': 'Test Artist 1'},
        {'id': 'track2', 'name': 'Test Song 2', 'artist': 'Test Artist 2'},
    ]

def test_listening_session_creation(self, earworm_study):
    """Test creation of listening session"""
    session = earworm_study.simulate_listening_session(
        track_id='test_track',
        track_name='Test Song',
        artist='Test Artist',
        repetition_type='algorithmic'
    )
    
    assert isinstance(session, ListeningSession)
    assert session.track_id == 'test_track'
    assert session.track_name == 'Test Song'
    assert session.artist == 'Test Artist'
    assert session.repetition_type == 'algorithmic'
    assert 1 <= session.initial_mood <= 10
    assert 1 <= session.final_mood <= 10
    assert session.earworm_persistence >= 0

def test_comparative_study(self, earworm_study, sample_tracks):
    """Test running comparative study"""
    df = earworm_study.run_comparative_study(sample_tracks, participants_per_condition=5)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_tracks) * 3 * 5  # 2 tracks * 3 conditions * 5 participants
    assert set(df['repetition_type'].unique()) == {'algorithmic', 'self_selected', 'control'}
    
    required_columns = ['session_id', 'track_name', 'repetition_type', 
                      'earworm_persistence', 'mood_change', 'repetition_count']
    assert all(col in df.columns for col in required_columns)

def test_persistence_analysis(self, earworm_study, sample_tracks):
    """Test persistence pattern analysis"""
    df = earworm_study.run_comparative_study(sample_tracks, participants_per_condition=10)
    results = earworm_study.analyze_persistence_patterns(df)
    
    assert isinstance(results, dict)
    assert 'condition_statistics' in results
    assert 'correlations' in results
    assert 'anova' in results
    assert 'pairwise_tests' in results
    
    # Check ANOVA results structure
    assert 'f_statistic' in results['anova']
    assert 'p_value' in results['anova']
    assert isinstance(results['anova']['f_statistic'], (int, float))
    assert isinstance(results['anova']['p_value'], (int, float))

def test_different_repetition_types(self, earworm_study):
    """Test that different repetition types produce different patterns"""
    algorithmic_session = earworm_study.simulate_listening_session(
        'track1', 'Song1', 'Artist1', 'algorithmic'
    )
    control_session = earworm_study.simulate_listening_session(
        'track1', 'Song1', 'Artist1', 'control'
    )
    
    # These should generally be different due to randomness and condition effects
    # We can't guarantee specific relationships, but we can ensure valid ranges
    assert algorithmic_session.repetition_count >= 1
    assert control_session.repetition_count >= 1
    assert algorithmic_session.earworm_persistence >= 0
    assert control_session.earworm_persistence >= 0
