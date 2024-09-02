import streamlit as st
import pandas as pd
import numpy as np

# importing the scripts
import scripts.spotify_api as spotify_api

# app title
st.title("Spotify Autoplaylists")

# Input field
playlist_ids = st.text_area("Enter Playlist IDs (comma-separated)", "37i9dQZF1DX9ND1QF5hZNF,37i9dQZF1DX1X7WV84927n")
# Convert the input into a list of IDs
playlist_ids = [pid.strip() for pid in playlist_ids.split(",")]

# Button to submit the input
if st.button("Analyze Playlists"):
    
    # fetching playlist data
    metadata_df = pd.DataFrame()
    tracks_df = pd.DataFrame()

    # # looping through all the playlists
    # for playlist_id in playlist_ids:
    #     # getting the metadata
    #     metadata = pd.DataFrame([spotify_api.get_playlist_metadata(playlist_id)])
    #     # saving 
    #     metadata_df = pd.concat([metadata_df, metadata])
    #     # getting the tracks
    #     track_list = spotify_api.get_playlist_tracks(playlist_id)
    #     # saving tracks
    #     tracks_df = pd.concat([tracks_df, track_list])

    # # getting the audio features
    # audio_features = spotify_api.get_audio_features(tracks_df['id'].values)

    # # merging the dataframes
    # tracks_df = tracks_df.merge(audio_features, on='id')

    # we 

    # displaying the playlist main information first
    st.write(f"Playlist Information")
    st.write(metadata_df[['playlist_id', 'playlist_name']].drop_duplicates())
