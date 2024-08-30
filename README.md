# Spotify Autoplaylists

Spotify Autoplaylists is a Python project that automatically curates and clusters Spotify tracks based on metadata to build personalized playlists. This tool helps to create musical experiences tailored to specific moods, genres, or other criteria.

## Features

- **Track Clustering:** Group tracks based on metadata like genre, mood, tempo, and more.
- **Automatic Playlist Creation:** Generate playlists automatically based on user-defined criteria.
- **Customizable:** Easily modify the clustering and playlist criteria to suit your preferences.

## Defining Your Playlists
Before running the project, you should define your selected playlists by creating a CSV file with the following structure.

```
playlist_name,playlist_id,link 
Electro chill,37i9dQZF1DX9ND1QF5hZNF
Hard Rock,37i9dQZF1DX1X7WV84927n
```
Create a file named `playlists.csv` in the `data/` directory with the following content:

### Columns:
- **playlist_name**: The name of the playlist for easier interpretation
- **playlist_id**: The Spotify playlist ID, which can be found in the URL of the playlist.


