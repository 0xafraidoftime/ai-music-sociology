import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(file), '..', 'src'))
from analysis.bias_detection import BiasAnalyzer, GeneratedSample
class TestBiasAnalyzer:
@pytest.fixture
def bias_analyzer(self):
    """Create BiasAnalyzer instance for testing"""
    return BiasAnalyzer()

def test_cultural_prompts_loading(self, bias_analyzer):
    """Test that cultural prompts are loaded correctly"""
    prompts = bias_analyzer.cultural_prompts
    
    assert isinstance(prompts, dict)
    assert 'western_contemporary' in prompts
    assert 'african' in prompts
    assert 'asian' in prompts
    
    # Check that each culture has multiple prompts
    for culture, culture_prompts in prompts.items():
        assert isinstance(culture_prompts, list)
        assert len(culture_prompts) > 0
        assert all(isinstance(prompt, str) for prompt in culture_prompts)

def test_music_generation_simulation(self, bias_analyzer):
    """Test simulated music generation"""
    sample = bias_analyzer.simulate_music_generation(
        prompt='traditional african drumming',
        cultural_context='african'
    )
    
    assert isinstance(sample, GeneratedSample)
    assert sample.prompt == 'traditional african drumming'
    assert sample.cultural_context == 'african'
    assert isinstance(sample.audio_features, dict)
    
    # Check that audio features are in reasonable ranges
    features = sample.audio_features
    assert 60 <= features['tempo'] <= 200
    assert 0 <= features['instrumentation_diversity'] <= 1
    assert 0 <= features['rhythmic_complexity'] <= 1

def test_biased_feature_generation(self, bias_analyzer):
    """Test that biased features show expected patterns"""
    # Generate samples for different cultural contexts
    western_sample = bias_analyzer.simulate_music_generation(
        'pop ballad', 'western_contemporary'
    )
    african_sample = bias_analyzer.simulate_music_generation(
        'traditional drumming', 'african'
    )
    
    # Features should be different (though we can't guarantee specific relationships)
    western_features = western_sample.audio_features
    african_features = african_sample.audio_features
    
    assert isinstance(western_features, dict)
    assert isinstance(african_features, dict)
    
    # Both should have same feature keys
    assert set(western_features.keys()) == set(african_features.keys())

def test_comprehensive_bias_test(self, bias_analyzer):
    """Test running comprehensive bias test"""
    # Limit to subset for faster testing
    bias_analyzer.cultural_prompts = {
        'western_contemporary': ['pop song', 'rock ballad'],
        'african': ['traditional drumming', 'afrobeat'],
    }
    
    df = bias_analyzer.run_comprehensive_bias_test()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    required_columns = ['sample_id', 'prompt', 'cultural_context', 'genre',
                      'tempo', 'instrumentation_diversity', 'rhythmic_complexity']
    assert all(col in df.columns for col in required_columns)
    
    # Should have samples from both cultures
    cultures = df['cultural_context'].unique()
    assert 'western_contemporary' in cultures
    assert 'african' in cultures

def test_bias_metrics_calculation(self, bias_analyzer):
    """Test bias metrics calculation"""
    # Create sample data
    data = {
        'cultural_context': ['western_contemporary'] * 10 + ['african'] * 10,
        'tempo': [120] * 10 + [140] * 10,
        'instrumentation_diversity': [0.5] * 10 + [0.6] * 10,
        'rhythmic_complexity': [0.4] * 10 + [0.7] * 10,
        'harmonic_complexity': [0.5] * 10 + [0.3] * 10,
        'dynamic_range': [0.6] * 10 + [0.5] * 10,
        'genre': ['pop'] * 10 + ['world'] * 10
    }
    df = pd.DataFrame(data)
    
    metrics = bias_analyzer.calculate_bias_metrics(df)
    
    assert isinstance(metrics, dict)
    assert 'cultural_feature_stats' in metrics
    assert 'tempo_diversity' in metrics
    assert 'western_vs_nonwestern_tests' in metrics
    assert 'bias_severity_scores' in metrics
    
    # Check bias severity scores
    bias_scores = metrics['bias_severity_scores']
    assert 'african' in bias_scores
    assert isinstance(bias_scores['african'], (int, float))
    assert bias_scores['african'] >= 0

def test_genre_inference(self, bias_analyzer):
    """Test genre inference from prompts"""
    test_cases = [
        ('pop ballad', 'pop'),
        ('rock anthem', 'rock'),
        ('electronic dance music', 'electronic'),
        ('traditional african music', 'world'),
        ('classical symphony', 'classical'),
    ]
    
    for prompt, expected_genre in test_cases:
        inferred_genre = bias_analyzer._infer_genre(prompt)
        assert inferred_genre == expected_genre

def test_diversity_metric(self, bias_analyzer):
    """Test diversity metric calculation"""
    # Create test data with known diversity
    df_high_diversity = pd.DataFrame({
        'cultural_context': ['a', 'b', 'c'],
        'tempo': [60, 120, 180]  # High diversity
    })
    
    df_low_diversity = pd.DataFrame({
        'cultural_context': ['a', 'b', 'c'],
        'tempo': [118, 120, 122]  # Low diversity
    })
    
    high_div = bias_analyzer._calculate_diversity_metric(df_high_diversity, 'tempo')
    low_div = bias_analyzer._calculate_diversity_metric(df_low_diversity, 'tempo')
    
    assert high_div > low_div
    assert high_div >= 0
    assert low_div >= 0
if name == 'main':
pytest.main([file])
