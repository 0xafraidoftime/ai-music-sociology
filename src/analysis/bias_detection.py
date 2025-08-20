import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests
import json
import time
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GeneratedSample:
    """Data class for storing generated music samples and metadata"""
    sample_id: str
    prompt: str
    cultural_context: str
    genre: str
    generation_service: str
    audio_features: Dict
    generated_at: str
    file_path: Optional[str] = None

class BiasAnalyzer:
    """
    Examines cultural, genre, and demographic biases in AI music generation tools
    by analyzing patterns in generated samples across diverse prompts.
    """
    
    def __init__(self):
        self.samples = []
        self.cultural_prompts = self._load_cultural_prompts()
        self.scaler = StandardScaler()
        
    def _load_cultural_prompts(self) -> Dict[str, List[str]]:
        """Load diverse cultural and genre prompts for testing"""
        return {
            'african': [
                'traditional african drumming',
                'west african highlife music',
                'south african jazz',
                'ethiopian traditional music',
                'afrobeat rhythms'
            ],
            'asian': [
                'traditional chinese erhu melody',
                'japanese koto music',
                'indian classical raga',
                'korean traditional music',
                'gamelan orchestra'
            ],
            'latin': [
                'brazilian bossa nova',
                'argentinian tango',
                'mexican mariachi',
                'cuban salsa',
                'reggaeton beat'
            ],
            'european': [
                'irish traditional folk',
                'spanish flamenco guitar',
                'russian folk dance',
                'french chanson',
                'german classical'
            ],
            'middle_eastern': [
                'arabic oud music',
                'persian classical',
                'turkish folk music',
                'israeli klezmer',
                'egyptian traditional'
            ],
            'western_contemporary': [
                'american pop ballad',
                'british rock anthem',
                'country music',
                'hip hop beat',
                'electronic dance music'
            ]
        }
    
    def simulate_music_generation(
        self, 
        prompt: str, 
        cultural_context: str,
        service: str = 'simulated'
    ) -> GeneratedSample:
        """
        Simulate music generation (replace with actual API calls in real implementation)
        """
        
        # Simulate realistic bias patterns in generated music
        features = self._generate_biased_features(cultural_context, prompt)
        
        sample = GeneratedSample(
            sample_id=f"sample_{len(self.samples)}",
            prompt=prompt,
            cultural_context=cultural_context,
            genre=self._infer_genre(prompt),
            generation_service=service,
            audio_features=features,
            generated_at=pd.Timestamp.now().isoformat()
        )
        
        self.samples.append(sample)
        return sample
    
    def _generate_biased_features(self, cultural_context: str, prompt: str) -> Dict:
        """
        Simulate realistic bias patterns that might appear in AI music generation.
        Based on known biases in training data and Western-centric datasets.
        """
        
        # Base features with Western pop bias (common in AI training data)
        base_features = {
            'tempo': 120,  # Standard pop tempo
            'key': 'C',    # Most common key in Western music
            'time_signature': '4/4',
            'duration': 180,  # 3 minutes
            'instrumentation_diversity': 0.5,
            'rhythmic_complexity': 0.4,
            'harmonic_complexity': 0.5,
            'melodic_range': 12,  # One octave
            'dynamic_range': 0.6
        }
        
        # Apply cultural adjustments (with bias patterns)
        if cultural_context == 'western_contemporary':
            # Least bias - this is what AI is trained on most
            adjustments = {
                'tempo': np.random.normal(0, 10),
                'instrumentation_diversity': np.random.normal(0, 0.1),
                'rhythmic_complexity': np.random.normal(0, 0.1)
            }
        elif cultural_context in ['african', 'latin']:
            # Moderate bias - some representation but stereotypical
            adjustments = {
                'tempo': np.random.normal(20, 15),  # Assumed to be "more rhythmic"
                'rhythmic_complexity': 0.7,  # Stereotypical assumption
                'instrumentation_diversity': np.random.normal(-0.1, 0.2)  # Limited variety
            }
        elif cultural_context in ['asian', 'middle_eastern']:
            # High bias - often reduced to pentatonic or modal characteristics
            adjustments = {
                'tempo': np.random.normal(-20, 20),
                'harmonic_complexity': 0.3,  # Oversimplified
                'melodic_range': 8,  # Pentatonic assumption
                'instrumentation_diversity': np.random.normal(-0.2, 0.1)
            }
        else:  # European traditional
            # Moderate bias toward classical structures
            adjustments = {
                'tempo': np.random.normal(-10, 20),
                'harmonic_complexity': 0.8,  # Classical bias
                'rhythmic_complexity': 0.3
            }
        
        # Apply adjustments
        for feature, adjustment in adjustments.items():
            if feature in base_features:
                if isinstance(base_features[feature], (int, float)):
                    base_features[feature] += adjustment
                    # Ensure reasonable bounds
                    if feature == 'tempo':
                        base_features[feature] = max(60, min(200, base_features[feature]))
                    elif feature.endswith('_complexity') or feature.endswith('_diversity'):
                        base_features[feature] = max(0, min(1, base_features[feature]))
        
        return base_features
    
    def _infer_genre(self, prompt: str) -> str:
        """Infer genre from prompt text"""
        genre_keywords = {
            'pop': ['pop', 'ballad', 'contemporary'],
            'rock': ['rock', 'anthem'],
            'electronic': ['electronic', 'edm', 'dance'],
            'jazz': ['jazz', 'swing'],
            'classical': ['classical', 'symphony'],
            'folk': ['folk', 'traditional'],
            'world': ['african', 'asian', 'arabic', 'indian', 'chinese']
        }
        
        prompt_lower = prompt.lower()
        for genre, keywords in genre_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return genre
        return 'other'
    
    def run_comprehensive_bias_test(self) -> pd.DataFrame:
        """Run comprehensive bias testing across all cultural contexts"""
        
        all_samples = []
        
        for cultural_context, prompts in self.cultural_prompts.items():
            for prompt in prompts:
                # Generate multiple samples per prompt to account for variation
                for i in range(3):  # 3 samples per prompt
                    sample = self.simulate_music_generation(
                        prompt=prompt,
                        cultural_context=cultural_context,
                        service='test_ai_generator'
                    )
                    all_samples.append(sample)
        
        # Convert to DataFrame for analysis
        df_samples = []
        for sample in all_samples:
            row = {
                'sample_id': sample.sample_id,
                'prompt': sample.prompt,
                'cultural_context': sample.cultural_context,
                'genre': sample.genre,
                'generation_service': sample.generation_service,
                **sample.audio_features
            }
            df_samples.append(row)
        
        return pd.DataFrame(df_samples)
    
    def calculate_bias_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate various bias metrics"""
        
        metrics = {}
        
        # 1. Feature distribution analysis
        feature_cols = ['tempo', 'instrumentation_diversity', 'rhythmic_complexity', 
                       'harmonic_complexity', 'dynamic_range']
        
        cultural_stats = df.groupby('cultural_context')[feature_cols].agg(['mean', 'std'])
        metrics['cultural_feature_stats'] = cultural_stats
        
        # 2. Diversity metrics
        metrics['tempo_diversity'] = self._calculate_diversity_metric(df, 'tempo')
        metrics['harmonic_diversity'] = self._calculate_diversity_metric(df, 'harmonic_complexity')
        
        # 3. Representation bias (how often does each culture appear in expected genres)
        genre_culture_crosstab = pd.crosstab(df['cultural_context'], df['genre'])
        metrics['genre_representation'] = genre_culture_crosstab
        
        # 4. Statistical significance tests
        from scipy import stats
        western = df[df['cultural_context'] == 'western_contemporary']
        non_western = df[df['cultural_context'] != 'western_contemporary']
        
        significance_tests = {}
        for feature in feature_cols:
            stat, p_value = stats.mannwhitneyu(
                western[feature], non_western[feature], alternative='two-sided'
            )
            significance_tests[feature] = {'statistic': stat, 'p_value': p_value}
        
        metrics['western_vs_nonwestern_tests'] = significance_tests
        
        # 5. Bias severity score (custom metric)
        bias_scores = {}
        for culture in df['cultural_context'].unique():
            if culture != 'western_contemporary':
                culture_subset = df[df['cultural_context'] == culture]
                western_subset = df[df['cultural_context'] == 'western_contemporary']
                
                # Calculate normalized differences
                differences = []
                for feature in feature_cols:
                    culture_mean = culture_subset[feature].mean()
                    western_mean = western_subset[feature].mean()
                    pooled_std = np.sqrt((culture_subset[feature].var() + western_subset[feature].var()) / 2)
                    normalized_diff = abs(culture_mean - western_mean) / pooled_std
                    differences.append(normalized_diff)
                
                bias_scores[culture] = np.mean(differences)
        
        metrics['bias_severity_scores'] = bias_scores
        
        return metrics
    
    def _calculate_diversity_metric(self, df: pd.DataFrame, feature: str) -> float:
        """Calculate diversity metric for a given feature across cultural contexts"""
        cultural_means = df.groupby('cultural_context')[feature].mean()
        return cultural_means.std() / cultural_means.mean()  # Coefficient of variation
    
    def visualize_bias_patterns(self, df: pd.DataFrame, metrics: Dict) -> None:
        """Create comprehensive visualizations of bias patterns"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Tempo distribution by culture
        sns.boxplot(data=df, x='cultural_context', y='tempo', ax=axes[0,0])
        axes[0,0].set_title('Tempo Distribution by Cultural Context')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Instrumentation diversity
        sns.boxplot(data=df, x='cultural_context', y='instrumentation_diversity', ax=axes[0,1])
        axes[0,1].set_title('Instrumentation Diversity by Cultural Context')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Rhythmic complexity
        sns.boxplot(data=df, x='cultural_context', y='rhythmic_complexity', ax=axes[1,0])
        axes[1,0].set_title('Rhythmic Complexity by Cultural Context')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Harmonic complexity
        sns.boxplot(data=df, x='cultural_context', y='harmonic_complexity', ax=axes[1,1])
        axes[1,1].set_title('Harmonic Complexity by Cultural Context')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 5. Genre distribution heatmap
        genre_culture = pd.crosstab(df['cultural_context'], df['genre'])
        sns.heatmap(genre_culture, annot=True, fmt='d', ax=axes[2,0], cmap='YlOrRd')
        axes[2,0].set_title('Genre Distribution by Cultural Context')
        
        # 6. Bias severity scores
        if 'bias_severity_scores' in metrics:
            cultures = list(metrics['bias_severity_scores'].keys())
            scores = list(metrics['bias_severity_scores'].values())
            axes[2,1].bar(cultures, scores)
            axes[2,1].set_title('Bias Severity Scores')
            axes[2,1].tick_params(axis='x', rotation=45)
            axes[2,1].set_ylabel('Normalized Difference from Western')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive Plotly dashboard for exploring bias patterns"""
        
        # Create PCA visualization
        feature_cols = ['tempo', 'instrumentation_diversity', 'rhythmic_complexity', 
                       'harmonic_complexity', 'dynamic_range']
        
        features = df[feature_cols].fillna(df[feature_cols].mean())
        scaled_features = self.scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(scaled_features)
        
        df_plot = df.copy()
        df_plot['PC1'] = pca_coords[:, 0]
        df_plot['PC2'] = pca_coords[:, 1]
        
        fig = px.scatter(
            df_plot,
            x='PC1', y='PC2',
            color='cultural_context',
            size='tempo',
            hover_data=['prompt', 'genre', 'harmonic_complexity'],
            title='AI Music Generation Bias Analysis - PCA Visualization'
        )
        
        fig.update_layout(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)",
            height=600
        )
        
        return fig
    
    def generate_bias_report(self, df: pd.DataFrame, metrics: Dict) -> str:
        """Generate comprehensive bias analysis report"""
        
        report = f"""
# AI Music Generation Bias Analysis Report

## Executive Summary
This report analyzes potential biases in AI music generation systems across {len(df['cultural_context'].unique())} cultural contexts and {len(df)} generated samples.

## Key Findings

### Cultural Representation
- Total Samples: {len(df)}
- Cultural Contexts Tested: {', '.join(df['cultural_context'].unique())}
- Genres Identified: {', '.join(df['genre'].unique())}

### Bias Severity Rankings
"""
        
        if 'bias_severity_scores' in metrics:
            sorted_bias = sorted(metrics['bias_severity_scores'].items(), 
                               key=lambda x: x[1], reverse=True)
            for i, (culture, score) in enumerate(sorted_bias, 1):
                report += f"{i}. {culture.replace('_', ' ').title()}: {score:.3f}\n"
        
        report += f"""

### Statistical Analysis
Western vs Non-Western Comparisons:
"""
        
        if 'western_vs_nonwestern_tests' in metrics:
            for feature, test_result in metrics['western_vs_nonwestern_tests'].items():
                significance = "***" if test_result['p_value'] < 0.001 else \
                             "**" if test_result['p_value'] < 0.01 else \
                             "*" if test_result['p_value'] < 0.05 else ""
                report += f"- {feature}: p={test_result['p_value']:.6f} {significance}\n"
        
        report += f"""

### Diversity Metrics
- Tempo Diversity (CV): {metrics.get('tempo_diversity', 0):.3f}
- Harmonic Diversity (CV): {metrics.get('harmonic_diversity', 0):.3f}

## Recommendations

### For AI Music Platform Developers
1. **Expand Training Data**: Include more diverse cultural music in training datasets
2. **Bias Testing**: Implement regular bias audits across cultural contexts  
3. **Cultural Consultation**: Work with ethnomusicologists and cultural experts
4. **User Controls**: Allow users to specify desired cultural authenticity levels

### For Researchers
1. **Longitudinal Studies**: Track bias patterns over time as models evolve
2. **Human Evaluation**: Complement automated analysis with cultural expert reviews
3. **Intersectional Analysis**: Examine biases across multiple dimensions (gender, age, etc.)

### For Policy Makers
1. **Algorithmic Auditing**: Require bias testing for commercial AI music systems
2. **Cultural Heritage Protection**: Consider implications for traditional music preservation
3. **Fair Representation**: Encourage funding for diverse music AI research

## Methodology Notes
- Simulated generation data used for demonstration purposes
- Real implementation would require actual API access to generation services
- Statistical tests assume normal distribution; non-parametric alternatives recommended
- Bias severity scores are relative measures, not absolute assessments

## Limitations
- Limited sample size per cultural context
- Simulation may not capture all real-world bias patterns
- Cultural contexts are broad generalizations
- No evaluation of perceived cultural authenticity by cultural practitioners

## Future Work
- Expand to more specific regional music traditions
- Include user preference and satisfaction metrics
- Develop automated bias detection algorithms
- Create standardized bias evaluation protocols for the music AI industry
"""
        
        return report
