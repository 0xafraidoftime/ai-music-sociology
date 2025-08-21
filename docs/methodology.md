Research Methodology
Overview
This document outlines the methodological approaches used in the AI & Music Sociology research project.
Emotional Affordances Mapping
Theoretical Framework
Based on Gibson's theory of affordances applied to musical experiences, examining how musical features create psychological and emotional "affordances" for listeners.
Data Sources

Spotify Web API: Audio features (valence, energy, danceability, etc.)
Genius API: Lyrical content for sentiment analysis
User Surveys: Subjective emotional responses (optional)

Analysis Methods
Audio Feature Analysis

Principal Component Analysis (PCA): Dimensionality reduction for visualization
K-means Clustering: Grouping songs by emotional characteristics
t-SNE: Non-linear dimensionality reduction for complex patterns

Natural Language Processing

VADER Sentiment Analysis: Rule-based sentiment scoring
TextBlob: Polarity and subjectivity analysis
Transformer Models: Deep learning-based emotion classification

Statistical Methods

ANOVA: Testing for group differences
Correlation Analysis: Feature relationships
Regression Models: Predicting emotional responses

Validation Approaches

Cross-validation: 5-fold cross-validation for model stability
Inter-rater Reliability: Multiple annotators for emotion labeling
External Validation: Comparison with existing music emotion datasets

2. Algorithmic Earworms Study
Experimental Design
Randomized controlled trial examining the effects of different recommendation algorithms on song persistence and mood.
Conditions

Algorithmic Condition: AI-curated song repetition (Spotify-style autoplay)
Self-Selected Condition: User-controlled repetition
Control Condition: No repetition/single play

Participants

Target: 60 participants (20 per condition)
Demographics: Age 18-65, diverse musical backgrounds
Recruitment: University subject pool, online platforms

Procedure

Pre-session Survey: Demographics, music preferences, current mood
Listening Session: 30-minute guided listening experience
Immediate Post-session: Mood assessment, persistence rating
Follow-up Surveys: 2, 6, 24 hours post-session

Measures

Earworm Persistence: Self-reported involuntary musical imagery
Mood Changes: 10-point Likert scales for various emotions
Listening Behavior: Repetition patterns, skip rates
Physiological Measures (optional): Heart rate, skin conductance

Statistical Analysis Plan

Mixed-effects Models: Account for repeated measures and individual differences
Survival Analysis: Time-to-earworm-cessation
Mediation Analysis: Explore mechanisms (mood → repetition → persistence)

3. AI Music Generation Bias Detection
Framework
Systematic evaluation of cultural, gender, and genre biases in AI music generation systems.
Target Systems

Suno AI: Text-to-music generation
AIVA: AI composition system
Jukebox: OpenAI's neural audio generation
Custom Models: Local implementations when APIs unavailable

Bias Dimensions

Cultural Bias: Western vs. non-Western musical traditions
Genre Bias: Representation across musical styles
Instrumentation Bias: Variety in generated instruments
Rhythmic Bias: Complexity and diversity of rhythmic patterns

Testing Protocol
Prompt Design

Systematic Sampling: 50 prompts per cultural context
Controlled Variables: Keep descriptive complexity constant
Cultural Authenticity: Prompts reviewed by cultural experts

Feature Extraction

Audio Analysis: Librosa for spectral, rhythmic, and harmonic features
Cultural Metrics: Custom algorithms for cultural music characteristics
Diversity Measures: Shannon entropy, Gini coefficient

Fairness Metrics

Demographic Parity: Equal representation across groups
Equalized Odds: Performance consistency across cultural contexts
Individual Fairness: Similar inputs → similar outputs

Evaluation Methods

Expert Review: Ethnomusicologists evaluate cultural authenticity
Listener Studies: Perception of cultural appropriateness
Algorithmic Auditing: Systematic bias detection algorithms

4. Data Management and Ethics
Data Privacy

Anonymization: Remove personally identifiable information
Secure Storage: Encrypted databases, limited access
Retention Policies: Data deletion after research completion

Ethical Considerations

IRB Approval: Institutional Review Board approval for human subjects research
Informed Consent: Clear explanation of research purposes and data use
Cultural Sensitivity: Respectful treatment of cultural music traditions
Platform Compliance: Adherence to API terms of service

Open Science Practices

Code Availability: All analysis code publicly available
Data Sharing: Anonymized datasets shared when possible
Replication: Detailed protocols for study replication
Pre-registration: Study protocols registered before data collection

5. Quality Control
Validity Threats

Selection Bias: Non-representative samples
Demand Characteristics: Participants changing behavior due to study participation
Technical Issues: API limitations, audio quality problems
Cultural Bias: Researcher perspectives influencing interpretation

Mitigation Strategies

Diverse Sampling: Active recruitment across demographic groups
Blinded Conditions: Participants unaware of specific hypotheses
Technical Validation: Regular checks of data quality and API reliability
Cultural Consultation: Collaboration with ethnomusicology experts

Statistical Power

Power Analysis: Minimum sample sizes calculated for each study
Effect Size Reporting: Practical significance alongside statistical significance
Multiple Comparisons: Appropriate corrections for family-wise error rates

6. Limitations
Methodological Limitations

Simulation vs. Reality: Some analyses use simulated data pending API access
Short-term Effects: Limited ability to study long-term behavioral changes
Platform Dependence: Results may be specific to particular music platforms
Cultural Generalizability: Western-centric research team and initial focus

Technical Limitations

API Constraints: Rate limits and data availability restrictions
Audio Quality: Compression and streaming artifacts in analysis
Model Interpretability: Black-box nature of some AI systems
Cross-platform Validity: Differences between music recommendation systems

References

Gibson, J. J. (1979). The Ecological Approach to Visual Perception
Juslin, P. N., & Västfjäll, D. (2008). Emotional responses to music: The need to consider underlying mechanisms. Behavioral and Brain Sciences
Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning
Casey, M. A., et al. (2008). Content-based music information retrieval: Current directions and future challenges. Proceedings of the IEEE

