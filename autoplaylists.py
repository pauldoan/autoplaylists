import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set the theme for better aesthetics
sns.set_theme(style="whitegrid")
# Create a custom color palette
palette = sns.color_palette("Set2")

# importing the scripts
import scripts.spotify_api as spotify_api

# app title
st.set_page_config(page_title="Spotify Autoplaylists", layout="wide")
st.title("Spotify Autoplaylists")

# Define the tabs
tabs = st.tabs(["Home", "Audio Feature Comparison", "Feature Histograms"])

# Tab 1: Input & Analyze
with tabs[0]:

    # Input field
    playlist_ids = st.text_area("Enter Playlist IDs (comma-separated)", "37i9dQZF1DX9ND1QF5hZNF,37i9dQZF1DX1X7WV84927n")
    playlist_ids = [pid.strip() for pid in playlist_ids.split(",")]


    # Button to submit the input
    if st.button("Analyze Playlists"):
    
# ---------------------------------------------------------------------------- #
#                         Fetching playlist information                       #
# ---------------------------------------------------------------------------- #

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

        # using sample datasets
        metadata_df = pd.read_csv("./data/streamlit_metadata_df.csv")
        tracks_df = pd.read_csv("./data/streamlit_tracks_df.csv")

        # merging the dataframes to get the playlist name
        tracks_df = tracks_df.merge(metadata_df[['playlist_name', 'playlist_id']], on='playlist_id')

        # Store the data in session state
        st.session_state['metadata_df'] = metadata_df
        st.session_state['tracks_df'] = tracks_df

        # Display the playlist metadata 
        st.subheader("Playlist Information")

        # Convert the image URLs to HTML image tags
        metadata_df['image'] = metadata_df['image'].apply(lambda x: f'<img src="{x}" width="60" height="60">')
        # Convert the playlist URLs to clickable links
        metadata_df['link'] = metadata_df['link'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')
        
        # Render the DataFrame in Streamlit with images
        metadata_df = metadata_df[['image', 'playlist_name', 'owner', 'description', 'followers', 'link']]
        st.markdown(
            metadata_df.to_html(escape=False, index=False), 
            unsafe_allow_html=True
        )

        # Done
        st.success("Analysis complete! Switch to other tabs to view results.")


# ---------------------------------------------------------------------------- #
#                           Display playlist metadata                         #
# ---------------------------------------------------------------------------- #


    


# ---------------------------------------------------------------------------- #
#                     Display playlist audio features stats                   #
# ---------------------------------------------------------------------------- #

# Tab 2: Audio Feature Comparison
with tabs[1]:
    if 'tracks_df' in st.session_state:
        st.header("Audio Feature Comparison Across Playlists")
        df_features = st.session_state['tracks_df'][['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'popularity', 'playlist_name']]
        df_melted = df_features.melt(id_vars='playlist_name', var_name='feature', value_name='value')

        # Create the FacetGrid plot
        g = sns.FacetGrid(df_melted, col="feature", col_wrap=3, height=4, aspect=1.5, sharey=False, hue='playlist_name', palette=palette)
        g.map(sns.barplot, "playlist_name", "value", order=None)

        # Add titles and improve the layout
        g.set_titles("{col_name}", size=16)
        g.set_axis_labels("Playlist", "Value")
        g.add_legend(title="Playlist Name", fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Render the plot in Streamlit
        st.pyplot(g.figure, dpi=200)
    else:
        st.warning("Please analyze the playlists first.")


# Tab 3: Feature Histograms
with tabs[2]:
    if 'tracks_df' in st.session_state:
        st.header("In-Playlist Feature Histograms")
        features = ['danceability', 'energy', 'popularity']

        for feature in features:
            st.subheader(f"Distribution of {feature.capitalize()} by Playlist")
            plt.figure(figsize=(10, 4))
            sns.histplot(data=st.session_state['tracks_df'], x=feature, hue='playlist_name', multiple='stack', palette=palette)
            plt.xlabel(feature.capitalize())
            plt.ylabel("Count")
            plt.title(f"{feature.capitalize()} Distribution by Playlist")
            st.pyplot(plt)
    else:
        st.warning("Please analyze the playlists first.")