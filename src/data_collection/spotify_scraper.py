import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import time
import requests

load_dotenv()

class SpotifyPlaylistAnalyzer:
    """
    Collects playlist data and audio features from Spotify Web API
    """
    
    def __init__(self):
        self.spotify = self._init_spotify_client()
        
    def _init_spotify_client(self):
        """Initialize Spotify API client with credentials"""
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=os.getenv('SPOTIFY_CLIENT_ID'),
                client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
            )
            return spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        except Exception as e:
            print(f"Error initializing Spotify client: {e}")
            print("Please ensure SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are set in your .env file")
            return None
    
    def get_playlist_tracks(self, playlist_id: str) -> pd.DataFrame:
        """Extract all tracks from a Spotify playlist"""
        if not self.spotify:
            raise ValueError("Spotify client not initialized")
        
        tracks_data = []
        offset = 0
        limit = 100
        
        while True:
            results = self.spotify.playlist_tracks(
                playlist_id, 
                offset=offset, 
                limit=limit,
                fields='items(track(id,name,artists,album,popularity,explicit,duration_ms,external_urls)),next'
            )
            
            for item in results['items']:
                if item['track'] and item['track']['id']:
                    track = item['track']
                    track_info = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                        'album_name': track['album']['name'],
                        'popularity': track['popularity'],
                        'explicit': track['explicit'],
                        'duration_ms': track['duration_ms'],
                        'spotify_url': track['external_urls']['spotify']
                    }
                    
                    # Add additional artists if present
                    if len(track['artists']) > 1:
                        track_info['all_artists'] = ', '.join([artist['name'] for artist in track['artists']])
                    else:
                        track_info['all_artists'] = track_info['artist_name']
                    
                    tracks_data.append(track_info)
            
            if not results['next']:
                break
            offset += limit
            
            # Rate limiting
            time.sleep(0.1)
        
        return pd.DataFrame(tracks_data)
    
    def get_audio_features(self, track_ids: List[str]) -> pd.DataFrame:
        """Get audio features for a list of track IDs"""
        if not self.spotify:
            raise ValueError("Spotify client not initialized")
        
        # Spotify API allows max 100 tracks per request
        batch_size = 100
        all_features = []
        
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i:i + batch_size]
            features = self.spotify.audio_features(batch_ids)
            
            # Filter out None values (tracks without audio features)
            valid_features = [f for f in features if f is not None]
            all_features.extend(valid_features)
            
            # Rate limiting
            time.sleep(0.1)
        
        return pd.DataFrame(all_features)
    
    def get_complete_playlist_analysis(self, playlist_id: str) -> pd.DataFrame:
        """Get complete playlist data including tracks and audio features"""
        
        # Get track information
        tracks_df = self.get_playlist_tracks(playlist_id)
        print(f"Retrieved {len(tracks_df)} tracks from playlist")
        
        if tracks_df.empty:
            return tracks_df
        
        # Get audio features
        track_ids = tracks_df['track_id'].tolist()
        audio_features_df = self.get_audio_features(track_ids)
        print(f"Retrieved audio features for {len(audio_features_df)} tracks")
        
        # Merge dataframes
        complete_df = tracks_df.merge(
            audio_features_df, 
            left_on='track_id', 
            right_on='id', 
            how='left'
        )
        
        # Clean up duplicate columns
        if 'id' in complete_df.columns:
            complete_df = complete_df.drop('id', axis=1)
        
        return complete_df
    
    def search_playlists(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for playlists by query"""
        if not self.spotify:
            raise ValueError("Spotify client not initialized")
        
        results = self.spotify.search(q=query, type='playlist', limit=limit)
        
        playlists = []
        for playlist in results['playlists']['items']:
            playlist_info = {
                'playlist_id': playlist['id'],
                'name': playlist['name'],
                'description': playlist['description'],
                'total_tracks': playlist['tracks']['total'],
                'owner': playlist['owner']['display_name'],
                'public': playlist['public'],
                'collaborative': playlist['collaborative'],
                'external_url': playlist['external_urls']['spotify']
            }
            playlists.append(playlist_info)
        
        return playlists
    
    def get_featured_playlists(self, country: str = 'US', limit: int = 20) -> List[Dict]:
        """Get featured playlists from Spotify"""
        if not self.spotify:
            raise ValueError("Spotify client not initialized")
        
        results = self.spotify.featured_playlists(country=country, limit=limit)
        
        playlists = []
        for playlist in results['playlists']['items']:
            playlist_info = {
                'playlist_id': playlist['id'],
                'name': playlist['name'],
                'description': playlist['description'],
                'total_tracks': playlist['tracks']['total'],
                'external_url': playlist['external_urls']['spotify']
            }
            playlists.append(playlist_info)
        
        return playlists
