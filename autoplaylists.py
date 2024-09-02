import streamlit as st

st.title("Spotify Playlist Audio Features Analyzer")

# Input field for playlist IDs
playlist_ids = st.text_area("Enter Playlist IDs (comma-separated)", "37i9dQZF1DX9ND1QF5hZNF,37i9dQZF1DX1X7WV84927n")

# Button to submit the input
if st.button("Analyze Playlists"):
    # Convert the input into a list of IDs
    playlist_ids = [pid.strip() for pid in playlist_ids.split(",")]

    # Here you'd include the logic to fetch and process the data
    # For now, let's display the playlist IDs
    st.write("You entered the following Playlist IDs:")
    st.write(playlist_ids)
