# Spotify Autoplaylists

Spotify Autoplaylists is an interactive web application that automatically retrieves and analyzes Spotify tracks based on audio features to help you build and curate playlists. This tool is a companion to create musical experiences tailored to specific moods, genres, or other criteria.

## Features

- **Home**: Get an overview of the app and provide playlist IDs to analyze
- **Track List**: View the tracks in the playlists with their audio features
- **Audio Feature Comparison**: Compare audio features accross playlists like energy, danceability, and more
- **Audio Feature Distribution**: Visualize the distribution of audio features within playlists
- **Statistical tests**: Perform statistical tests to compare audio features across playlists (COMING SOON)
- **Modeling**: Build, train and tune a Deep Learning model to understand the audio features and predict the ideal playlist for a track
- **Recommendations**: Get playlist recommendations for a unseen track using the trained model
- **Playlist Management**: Curate a playlist by deciding whether to add a new tracks based on its features


## Defining Your Playlists
Before running the project, you should define your selected playlists by creating a CSV file with the following structure.

```
playlist_name,playlist_id 
Electro chill,37i9dQZF1DX9ND1QF5hZNF
Hard Rock,37i9dQZF1DX1X7WV84927n
```
Create a file named `playlists.csv` in the `data/` directory with the following content:

### Columns:
- **playlist_name**: The name of the playlist for easier interpretation
- **playlist_id**: The Spotify playlist ID, which can be found in the URL of the playlist.


