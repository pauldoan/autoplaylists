# import libraries
from dotenv import load_dotenv
import os
import pandas as pd
import requests
import time
from tqdm.auto import tqdm

# Load environment variables from .env file
load_dotenv()

# Access the API key
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

# request access token
url = "https://accounts.spotify.com/api/token"
payload = {
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret
}
headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}
response = requests.post(url, data=payload, headers=headers)
response.raise_for_status()  # Raise an exception for HTTP errors
access_token = response.json()["access_token"]



# ---------------------------------------------------------------------------- #
#                                Tracks metatada                               #
# ---------------------------------------------------------------------------- #


# function to get hit Spotfy API and get playlist metadata
def get_tracks_metadata(track_ids, access_token=access_token):
    """
    Fetch metadata for a Spotify tracks.
    
    Parameters:
    - track_ids: A list or array of Spotify track IDs.
    - access_token: Your Spotify API Bearer token

    Returns:
    - A dataframe containing the tracks metadata
    """
    endpoint = f"https://api.spotify.com/v1/tracks"
    headers = {'Authorization': f'Bearer {access_token}'}
    
    # Initialize a list to hold all tracks data
    all_tracks = []
    
    # Spotify API allows max 100 IDs per request, so we process in batches
    batch_size = 50
    if isinstance(track_ids, str):
        track_ids = [track_ids]

    # looping through the track_ids in batches 
    for i in range(0, len(track_ids), batch_size):
        batch_ids = track_ids[i:i + batch_size]
        ids = ','.join(batch_ids)
        params = {'ids': ids}
        
        try:
            # Make a GET request to the audio features endpoint
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            tracks_meta = response.json()
            all_tracks.extend(tracks_meta['tracks'])

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            continue

    # Convert the list of tracks to a DataFrame
    all_tracks = pd.DataFrame(all_tracks)
    
    # dropping unnecessary columns
    all_tracks = all_tracks[
        [
            'id', 'name', 'album', 'artists', 'available_markets', 'disc_number', 'duration_ms', 'explicit', 
            'external_ids', 'external_urls', 'is_local', 'popularity', 'preview_url', 'track_number', 'type', 'uri'
        ]].reset_index(drop=True)
    
    return all_tracks


# ---------------------------------------------------------------------------- #
#                               PLaylist Metadata                              #
# ---------------------------------------------------------------------------- #


# function to get hit Spotfy API and get playlist metadata
def get_playlist_metadata(playlist_id, access_token=access_token):
    """
    Fetch metadata for a Spotify playlist.
    
    Parameters:
    - playlist_id: The Spotify playlist ID
    - access_token: Your Spotify API Bearer token

    Returns:
    - A dictionary containing the playlist metadata
    """
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
    headers = {'Authorization': f'Bearer {access_token}'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Extract relevant metadata
        metadata = {
            "playlist_name": data["name"],
            "playlist_id": data["id"],
            "owner": data["owner"]["display_name"],
            "description": data["description"],
            "followers": data["followers"]["total"],
            "link": data["external_urls"]["spotify"],
            "image": data["images"][0]["url"],

        }
        
        return metadata
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


# ---------------------------------------------------------------------------- #
#                              Playlist track list                             #
# ---------------------------------------------------------------------------- #

# function to get hit Spotfy API and get playlist tracks
def get_playlist_tracks(playlist_id, access_token=access_token):
    """
    Fetch all tracks from a Spotify playlist.
    
    Parameters:
    - playlist_id: The Spotify playlist ID
    - access_token: Your Spotify API Bearer token

    Returns:
    - A dataframe of all tracks in the playlist
    """
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {'Authorization': f'Bearer {access_token}'}
    
    track_list = []
    params = {
        'limit': 50,
        'offset': 0
        }
    
    while True:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            # Extract the 'track' value from each item
            tracks_data = [track['track'] for track in data['items'] if track['track'] is not None]

            # Extract track items from the response
            track_list.extend(tracks_data)

            # Check if there are more tracks to fetch
            if data['next']:
                params['offset'] += params['limit']
                time.sleep(1)  # To avoid hitting rate limits
            else:
                break

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            break
    
    # Create a DataFrame
    df = pd.DataFrame(track_list)

    # adding the playlist id to the dataframe
    df['playlist_id'] = playlist_id
    
    return df


# ---------------------------------------------------------------------------- #
#                             Tracks audio features                            #
# ---------------------------------------------------------------------------- #

def get_audio_features(track_ids, access_token=access_token):
    """
    Fetch audio features for a list of Spotify track IDs and return them as a DataFrame.
    
    Parameters:
    - track_ids: A list or array of Spotify track IDs.
    - access_token: Your Spotify API Bearer token.
    
    Returns:
    - A pandas DataFrame with track IDs as the index and audio features as columns.
    """
    endpoint = "https://api.spotify.com/v1/audio-features"
    headers = {'Authorization': f'Bearer {access_token}'}

    # Initialize a list to hold all audio feature data
    all_audio_features = []
    
    # Spotify API allows max 100 IDs per request, so we process in batches
    batch_size = 100
    if isinstance(track_ids, str):
        track_ids = [track_ids]

    # looping through the track_ids in batches 
    for i in range(0, len(track_ids), batch_size):
        batch_ids = track_ids[i:i + batch_size]
        ids = ','.join(batch_ids)
        params = {'ids': ids}
        
        try:
            # Make a GET request to the audio features endpoint
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            audio_features = response.json()['audio_features']
            all_audio_features.extend(audio_features)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            continue

    # Convert the list of audio features to a DataFrame
    audio_features_df = pd.DataFrame(all_audio_features)
    
    # dropping unnecessary columns
    audio_features_df.drop(['type', 'uri', 'track_href', 'analysis_url', 'duration_ms'], axis=1, inplace=True)
    
    return audio_features_df.reset_index()