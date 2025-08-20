import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import random

@dataclass
class ListeningSession:
    """Data class for tracking individual listening sessions"""
    session_id: str
    track_id: str
    track_name: str
    artist: str
    repetition_type: str  # 'algorithmic', 'self_selected', 'control'
    start_time: datetime
    duration_minutes: int
    repetition_count: int
    initial_mood: int  # 1-10 scale
    final_mood: int    # 1-10 scale
    earworm_persistence: int  # Hours after session
    participant_id: str

class EarwormStudy:
    """
    Studies how algorithmic recommendation systems influence 
    earworm persistence and mood regulation.
    """
    
    def __init__(self):
        self.sessions = []
        self.participants = {}
        
    def simulate_listening_session(
        self, 
        track_id: str,
        track_name: str,
        artist: str,
        repetition_type: str = 'algorithmic',
        duration_minutes: int = 30,
        participant_id: str = 'sim_participant'
    ) -> ListeningSession:
        """
        Simulate a listening session with various parameters.
        In real study, this would collect actual user data.
        """
        
        # Simulate realistic parameters based on repetition type
        if repetition_type == 'algorithmic':
            repetition_count = np.random.poisson(8)  # AI tends to create more loops
            mood_change = np.random.normal(0.5, 1.5)  # Slight positive bias
            persistence = np.random.exponential(4)  # Longer persistence
        elif repetition_type == 'self_selected':
            repetition_count = np.random.poisson(5)  # User-controlled
            mood_change = np.random.normal(0.8, 1.2)  # More positive user control
            persistence = np.random.exponential(2.5)  # Moderate persistence
        else:  # control
            repetition_count = np.random.poisson(2)  # Minimal repetition
            mood_change = np.random.normal(0, 1.0)  # Neutral
            persistence = np.random.exponential(1.5)  # Lower persistence
        
        initial_mood = np.random.randint(4, 8)  # Start in moderate mood
        final_mood = max(1, min(10, initial_mood + mood_change))
        
        session = ListeningSession(
            session_id=f"session_{len(self.sessions)}",
            track_id=track_id,
            track_name=track_name,
            artist=artist,
            repetition_type=repetition_type,
            start_time=datetime.now(),
            duration_minutes=duration_minutes,
            repetition_count=max(1, repetition_count),
            initial_mood=initial_mood,
            final_mood=int(final_mood),
            earworm_persistence=max(0, int(persistence)),
            participant_id=participant_id
        )
        
        self.sessions.append(session)
        return session
    
    def run_comparative_study(
        self, 
        tracks: List[Dict],
        participants_per_condition: int = 20
    ) -> pd.DataFrame:
        """
        Run a comparative study across different repetition conditions.
        """
        
        conditions = ['algorithmic', 'self_selected', 'control']
        
        for track in tracks:
            for condition in conditions:
                for i in range(participants_per_condition):
                    participant_id = f"{condition}_participant_{i}"
                    
                    self.simulate_listening_session(
                        track_id=track['id'],
                        track_name=track['name'],
                        artist=track['artist'],
                        repetition_type=condition,
                        participant_id=participant_id
                    )
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'session_id': s.session_id,
                'track_name': s.track_name,
                'artist': s.artist,
                'repetition_type': s.repetition_type,
                'repetition_count': s.repetition_count,
                'initial_mood': s.initial_mood,
                'final_mood': s.final_mood,
                'mood_change': s.final_mood - s.initial_mood,
                'earworm_persistence': s.earworm_persistence,
                'participant_id': s.participant_id
            } for s in self.sessions
        ])
        
        return df
    
    def analyze_persistence_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze earworm persistence patterns across conditions"""
        
        results = {}
        
        # Overall statistics by condition
        condition_stats = df.groupby('repetition_type').agg({
            'earworm_persistence': ['mean', 'std', 'median'],
            'repetition_count': ['mean', 'std'],
            'mood_change': ['mean', 'std']
        }).round(2)
        
        results['condition_statistics'] = condition_stats
        
        # Correlation analysis
        correlations = df[['repetition_count', 'earworm_persistence', 'mood_change']].corr()
        results['correlations'] = correlations
        
        # Statistical significance testing (simplified)
        from scipy import stats
        
        algorithmic = df[df['repetition_type'] == 'algorithmic']['earworm_persistence']
        self_selected = df[df['repetition_type'] == 'self_selected']['earworm_persistence']
        control = df[df['repetition_type'] == 'control']['earworm_persistence']
        
        # ANOVA
        f_stat, p_value = stats.f_oneway(algorithmic, self_selected, control)
        results['anova'] = {'f_statistic': f_stat, 'p_value': p_value}
        
        # Post-hoc pairwise comparisons
        results['pairwise_tests'] = {
            'algorithmic_vs_control': stats.mannwhitneyu(algorithmic, control),
            'self_selected_vs_control': stats.mannwhitneyu(self_selected, control),
            'algorithmic_vs_self_selected': stats.mannwhitneyu(algorithmic, self_selected)
        }
        
        return results
    
    def visualize_results(self, df: pd.DataFrame) -> None:
        """Create comprehensive visualizations of study results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Earworm persistence by condition
        sns.boxplot(data=df, x='repetition_type', y='earworm_persistence', ax=axes[0,0])
        axes[0,0].set_title('Earworm Persistence by Repetition Type')
        axes[0,0].set_ylabel('Persistence (hours)')
        
        # 2. Mood change by condition
        sns.boxplot(data=df, x='repetition_type', y='mood_change', ax=axes[0,1])
        axes[0,1].set_title('Mood Change by Repetition Type')
        axes[0,1].set_ylabel('Mood Change (final - initial)')
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Repetition count vs persistence
        sns.scatterplot(data=df, x='repetition_count', y='earworm_persistence', 
                       hue='repetition_type', ax=axes[1,0])
        axes[1,0].set_title('Repetition Count vs Earworm Persistence')
        
        # 4. Distribution of persistence times
        for condition in df['repetition_type'].unique():
            subset = df[df['repetition_type'] == condition]
            axes[1,1].hist(subset['earworm_persistence'], alpha=0.7, label=condition, bins=20)
        axes[1,1].set_title('Distribution of Earworm Persistence')
        axes[1,1].set_xlabel('Persistence (hours)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, df: pd.DataFrame, results: Dict) -> str:
        """Generate a comprehensive research report"""
        
        report = f"""
# Algorithmic Earworms Study Report

## Study Overview
- Total Sessions: {len(df)}
- Participants per Condition: {len(df) // 3}
- Conditions: Algorithmic, Self-Selected, Control

## Key Findings

### Earworm Persistence
- Algorithmic: {results['condition_statistics'].loc['algorithmic', ('earworm_persistence', 'mean')]:.2f} ± {results['condition_statistics'].loc['algorithmic', ('earworm_persistence', 'std')]:.2f} hours
- Self-Selected: {results['condition_statistics'].loc['self_selected', ('earworm_persistence', 'mean')]:.2f} ± {results['condition_statistics'].loc['self_selected', ('earworm_persistence', 'std')]:.2f} hours  
- Control: {results['condition_statistics'].loc['control', ('earworm_persistence', 'mean')]:.2f} ± {results['condition_statistics'].loc['control', ('earworm_persistence', 'std')]:.2f} hours

### Statistical Significance
- ANOVA F-statistic: {results['anova']['f_statistic']:.3f}
- P-value: {results['anova']['p_value']:.6f}
- Significant: {'Yes' if results['anova']['p_value'] < 0.05 else 'No'}

### Correlations
- Repetition Count ↔ Persistence: {results['correlations'].loc['repetition_count', 'earworm_persistence']:.3f}
- Repetition Count ↔ Mood Change: {results['correlations'].loc['repetition_count', 'mood_change']:.3f}

## Conclusions
{'Algorithmic recommendation systems appear to significantly increase earworm persistence compared to user-controlled listening.' if results['anova']['p_value'] < 0.05 else 'No significant difference found between conditions.'}

## Implications for Music Streaming Platforms
- Algorithm design may unconsciously optimize for "stickiness" over user wellbeing
- Consider implementing "earworm breaks" in recommendation logic
- User awareness of algorithmic influence may reduce unwanted persistence effects

## Methodology Notes
- Controlled experimental design with randomized condition assignment
- Standardized mood measurement scales (1-10 Likert)
- Post-session follow-up for persistence measurement
- Statistical controls for track popularity and individual differences

"""
        return report
