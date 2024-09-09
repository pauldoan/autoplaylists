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
import scripts.modeling as modeling

# app title
st.set_page_config(page_title="Spotify Autoplaylists", layout="wide")
st.title("Spotify Autoplaylists 🎧")

# Define the tabs
tabs = st.tabs(
    [
        "Home",
        "Track list",
        "Audio Feature Comparison",
        "Audio Feature Distribution",
        "Modeling",
        "Recommendations",
        "Playlist Management",
    ]
)


# ---------------------------------------------------------------------------- #
#                         Fetching playlist information                       #
# ---------------------------------------------------------------------------- #

with tabs[0]:

    st.subheader("Welcome to the Spotify Autoplaylists App")

    # Add a brief description of the app
    st.markdown(
        """
    **Spotify Autoplaylists** is an interactive web application that helps you analyze and manage playlists, like a Curator, based on tracks audio features.
    Once you provide playlists, the app will fetch the playlists data and track lists, analyze it, and provide insights!
    
    **Key Features:**
    - **Home**: Get an overview of the app and provide playlist IDs to analyze
    - **Track List**: View the tracks in the playlists with their audio features
    - **Audio Feature Comparison**: Compare audio features accross playlists like energy, danceability, and more
    - **Audio Feature Distribution**: Visualize the distribution of audio features within playlists
    - **Statistical tests**: Perform statistical tests to compare audio features across playlists (COMING SOON)
    - **Modeling**: Build, train and tune a Deep Learning model to understand the audio features and predict the ideal playlist for a track
    - **Recommendations**: Get playlist recommendations for a unseen track using the trained model
    - **Playlist Management**: Curate a playlist by deciding whether to add a new tracks based on its features

    Simply provide the playlist IDs, click on `Analyze`, and the app will do the rest!
    """
    )

    # Input field
    playlist_ids = st.text_area(
        "Enter Playlist IDs (comma-separated)",
        "37i9dQZF1DX9ND1QF5hZNF,37i9dQZF1DX1X7WV84927n,37i9dQZF1DX9sQDbOMReFI,37i9dQZF1DX1lVhptIYRda,37i9dQZF1DWTKxc7ZObqeH",
    )
    playlist_ids = [pid.strip() for pid in playlist_ids.split(",")]

    # Button to submit the input
    if st.button("Analyze Playlists 🎵"):

        with st.spinner("Fetching playlist data from Spotify... 🚀"):

            # fetching playlist data
            metadata_df = pd.DataFrame()
            tracks_df = pd.DataFrame()

            # looping through all the playlists
            for playlist_id in playlist_ids:
                # getting the metadata
                metadata = pd.DataFrame([spotify_api.get_playlist_metadata(playlist_id)])
                # saving
                metadata_df = pd.concat([metadata_df, metadata])
                # getting the tracks
                track_list = spotify_api.get_playlist_tracks(playlist_id)
                # saving tracks
                tracks_df = pd.concat([tracks_df, track_list])

            # getting the audio features
            audio_features = spotify_api.get_audio_features(tracks_df["id"].values)

            # merging the dataframes
            tracks_df = tracks_df.merge(audio_features, on="id")

            # merging the dataframes to get the playlist name
            tracks_df = tracks_df.merge(metadata_df[["playlist_name", "playlist_id"]], on="playlist_id")

            # Store the data in session state
            st.session_state["metadata_df"] = metadata_df
            st.session_state["tracks_df"] = tracks_df

        # Display the playlist metadata
        st.subheader("Playlist Information")

        # Convert the image URLs to HTML image tags
        metadata_df["image"] = metadata_df["image"].apply(lambda x: f'<img src="{x}" width="60" height="60">')
        # Convert the playlist URLs to clickable links
        metadata_df["link"] = metadata_df["link"].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')

        # Render the DataFrame in Streamlit with images
        metadata_df = metadata_df[["image", "playlist_name", "owner", "description", "followers", "link"]]
        st.markdown(metadata_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Done
        st.success("Analysis complete! Switch to other tabs to view results.")

    # Check if metadata exists in session state and display it
    elif "metadata_df" in st.session_state:
        st.subheader("Playlist Information")

        # Get the saved metadata from session state
        metadata_df = st.session_state["metadata_df"]

        # Convert the image URLs to HTML image tags
        metadata_df["image"] = metadata_df["image"].apply(lambda x: f'<img src="{x}" width="60" height="60">')
        # Convert the playlist URLs to clickable links
        metadata_df["link"] = metadata_df["link"].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')

        # Render the DataFrame in Streamlit with images
        metadata_df = metadata_df[["image", "playlist_name", "owner", "description", "followers", "link"]]
        st.markdown(metadata_df.to_html(escape=False, index=False), unsafe_allow_html=True)


# ---------------------------------------------------------------------------- #
#                                  Track list                                  #
# ---------------------------------------------------------------------------- #

with tabs[1]:

    if "tracks_df" in st.session_state:
        st.header("Track List")

        # streamlist select box for selecting playlist
        selected_playlist = st.selectbox("Select a Playlist", st.session_state["tracks_df"]["playlist_name"].unique())

        # filter the tracks
        tracks = st.session_state["tracks_df"][st.session_state["tracks_df"]["playlist_name"] == selected_playlist]

        # extract artist name for easier reading
        tracks["artist"] = tracks["artists"].apply(lambda x: x[0]["name"])

        feature_options = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
        ]

        # filter dataframe for better display
        tracks = tracks[["id", "name", "artist", "duration_ms", "popularity"] + feature_options]

        # display the track list
        st.dataframe(tracks)

    else:
        st.warning("Please analyze the playlists first.")


# ---------------------------------------------------------------------------- #
#                     Display playlist audio features comparison               #
# ---------------------------------------------------------------------------- #

with tabs[2]:
    if "tracks_df" in st.session_state:
        st.header("Audio Feature Comparison Across Playlists")

        st.markdown("""In this section, you can compare the audio features of the tracks across different playlists.""")

        # Define available features for the multiselect menu
        feature_options = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "popularity",
        ]

        # Streamlit multiselect menu for selecting features
        selected_features = st.multiselect(
            "Select features to analyze", feature_options, default=["energy", "danceability", "loudness"]
        )

        # Check if any features are selected
        if selected_features:
            # Filter DataFrame
            df_features = st.session_state["tracks_df"][selected_features + ["playlist_name"]]

            # Create columns for better layout
            cols = st.columns(3)

            # Loop through each selected feature and create a barplot
            for idx, feature in enumerate(selected_features):
                with cols[idx % 3]:

                    # Create plot per feature
                    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
                    sns.barplot(data=df_features, y="playlist_name", x=feature, palette="muted", ax=ax)

                    # Titles and labels
                    ax.set_xlabel(f"{feature.capitalize()}")
                    ax.set_ylabel("")
                    plt.tight_layout()

                    # Render the plot in Streamlit
                    st.pyplot(fig)

    else:
        st.warning("Please analyze the playlists first.")


# ---------------------------------------------------------------------------- #
#                             Features distribution                            #
# ---------------------------------------------------------------------------- #


with tabs[3]:
    if "tracks_df" in st.session_state:
        st.header("In-Playlist Feature Histograms")

        st.markdown("""In this section, you can visualize the distrbution of audio features within playlists.""")

        # allow user to select the feature with multiselect menu on streamlit
        features_options = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "popularity",
        ]
        # build streamlit multi-select menu
        features = st.multiselect(
            "Select features to analyze", features_options, default=["acousticness", "energy", "valence"]
        )
        # Create columns for better layout
        cols = st.columns(3)

        for idx, feature in enumerate(features):
            with cols[idx % 3]:

                # Create a responsive plot
                fig, ax = plt.subplots(figsize=(6, 4), dpi=150)  # Smaller figure size for more responsive plots
                sns.histplot(
                    data=st.session_state["tracks_df"],
                    x=feature,
                    hue="playlist_name",
                    multiple="layer",
                    palette="muted",
                    ax=ax,
                    bins=10,
                )

                # labels and title
                ax.set_xlabel(feature.capitalize())
                ax.set_ylabel("Count")
                ax.set_title(f"{feature.capitalize()} Distribution by Playlist")

                # Tighter layout for adaptive margins
                plt.tight_layout()

                # Render the plot in Streamlit
                st.pyplot(fig)
    else:
        st.warning("Please analyze the playlists first.")


# ---------------------------------------------------------------------------- #
#                                   Modeling                                   #
# ---------------------------------------------------------------------------- #

with tabs[4]:
    if "tracks_df" in st.session_state:
        st.header("Playlist features modeling")

        # adding a little description explaning that the model will be a Dense Neural Network with dropout
        st.markdown(
            """
        In this section, you can build a Deep Learning model to predict the playlist based on the audio features of the tracks.
        The model will be a simple Dense Neural Network with dropout layers to prevent overfitting.
        The model will use ReLU activation functions for the hidden layers.
        The model will be trained on the audio features of the tracks to predict the playlist.
        """
        )

        # streamlit input for hidden layer sizes
        hidden_sizes = st.text_input("Enter hidden layer sizes", "8,8")
        # convert to list of integers
        hidden_sizes = [int(size) for size in hidden_sizes.split(",")]

        # streamlit slider for the dropout rate
        dropout = st.slider("Select dropout rate", 0.0, 0.5, 0.1)

        # streamlit int slider for epochs
        epochs = st.slider("Select number of epochs", 1, 50, 10)

        # streamlist slider for learning rate
        lr = st.slider("Select learning rate", 0.001, 0.1, 0.01)

        # Initialize a flag to check if the model has been trained
        if "model_trained" not in st.session_state:
            st.session_state["model_trained"] = False

        # Handle the button for building/retraining the model
        button_label = "Retrain Model" if st.session_state["model_trained"] else "Build and Train Model"

        if st.button(button_label):

            with st.spinner("Building & Training Neural Network... 🤖"):
                # Build and train/retrain the model
                model, label_encoder, scaler, history, class_report = modeling.build_train(
                    st.session_state["tracks_df"],
                    epochs=epochs,
                    hidden_sizes=hidden_sizes,
                    lr=lr,
                    dropout=dropout,
                )

                # Store the model and other objects in session state
                st.session_state["model"] = model
                st.session_state["label_encoder"] = label_encoder
                st.session_state["scaler"] = scaler
                st.session_state["model_trained"] = True  # Update flag once model is trained

                # Success message
                st.success(f"{button_label} completed successfully!")

                # Print the main performance metrics
                st.subheader("Classification Report")
                accuracy = round(class_report.loc[["accuracy"], :].iloc[:, 0].values[0], 3)
                weighted_f1 = round(class_report.loc[["weighted avg"], "f1-score"].values[0], 3)
                st.write(f"Model Accuracy: {accuracy}")
                st.write(f"Weighted F1 Score: {weighted_f1}")
                st.dataframe(class_report)

                # Plot the training history using two columns
                st.subheader("Model Loss")
                # Create two columns
                col1, col2 = st.columns(2)

                # extracting training history
                train_loss_hist = history["train_loss"]
                val_loss_hist = history["valid_loss"]
                train_acc_hist = history["train_acc"]
                val_acc_hist = history["valid_acc"]

                with col1:
                    # Plot only the first axis (training loss)
                    fig_train, ax_train = plt.subplots()
                    sns.lineplot(train_loss_hist, ax=ax_train, label="Training loss")
                    sns.lineplot(val_loss_hist, ax=ax_train, label="Validation loss")
                    ax_train.set_title("Loss")
                    ax_train.legend()
                    st.pyplot(fig_train)  # Render training loss

                with col2:
                    # Plot only the second axis (validation loss)
                    fig_val, ax_val = plt.subplots()
                    sns.lineplot(train_acc_hist, ax=ax_val, label="Training Acc")
                    sns.lineplot(val_acc_hist, ax=ax_val, label="Validation Acc")
                    ax_val.set_title("Accuracy")
                    ax_val.legend()

                    st.pyplot(fig_val)  # Render validation loss

    else:
        st.warning("Please analyze the playlists first.")


# ---------------------------------------------------------------------------- #
#                                   Recommendations                           #
# ---------------------------------------------------------------------------- #

with tabs[5]:
    if "model" in st.session_state:
        st.header("Playlist features modeling")

        st.markdown(
            """This section allows you to get playlist recommendations for a track based on its audio features."""
        )

        # streamlit input for track ids
        track_ids = st.text_input(
            "Enter Track IDs (comma-separated)",
            "2zYzyRzz6pRmhPzyfMEC8s,6oanIhkNbxXnX19RTtkpEL,7KwZNVEaqikRSBSpyhXK2j",
            key=0,
        )

        # get tracks meta
        track_meta = spotify_api.get_tracks_metadata(track_ids)

        # hit spotify API
        af = spotify_api.get_audio_features(track_ids)
        # merge metadata and audio features
        track_df = pd.merge(track_meta, af, on="id")
        # extract artist name for easier reading
        track_df["artist"] = track_df["artists"].apply(lambda x: x[0]["name"])

        # streamlit button for getting recommendations
        if st.button("Get Recommendations"):
            # get the recommendations
            recommendations = modeling.model_inference(
                track_df, st.session_state["model"], st.session_state["scaler"], st.session_state["label_encoder"]
            )

            # looping trhough track ids
            for track_id in recommendations.id.unique():

                # extract data from recommendations dataframe
                track_recommendation = recommendations[recommendations["id"] == track_id]

                # subheader with track name and artist
                st.subheader(f"{track_recommendation['name'].values[0]} by {track_recommendation['artist'].values[0]}")

                # extracting dict of playlist and probabilities
                prob_dict = track_recommendation["all_probabilities"].values[0]

                col = st.columns(4)

                with col[0]:
                    # Embed the audio preview
                    preview_url = track_recommendation["preview_url"].values[0]
                    if pd.notna(preview_url):  # Ensure the URL is not NaN
                        st.audio(preview_url)
                    else:
                        st.write("No audio preview available for this track.")

                    # display recommendation
                    st.write(
                        "Best Playlist recommendation:", "`", track_recommendation["predicted_label"].values[0], "`"
                    )
                    st.write(
                        "Probability score:",
                        round(track_recommendation["predicted_probability"].values[0] * 100, 2),
                        "%",
                    )
                with col[1]:
                    fig, ax = plt.subplots()
                    sns.barplot(x=list(prob_dict.keys()), y=list(prob_dict.values()), ax=ax, palette=palette)
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)

    else:
        st.warning("Please train a model first.")


# ---------------------------------------------------------------------------- #
#                                   Playlist Management                        #
# ---------------------------------------------------------------------------- #

with tabs[6]:
    if "model" in st.session_state:
        st.header("Playlist Management")

        st.markdown(
            """This section allows you to curate a playlist by deciding whether to include new tracks based on their audio features."""
        )

        # streamlit input for selecting a target playlist
        target_playlist = st.selectbox(
            "Select a target playlist", st.session_state["tracks_df"]["playlist_name"].unique()
        )

        # streamlit input for track ids
        track_ids_include = st.text_input(
            "Enter Track IDs (comma-separated)",
            "2zYzyRzz6pRmhPzyfMEC8s,6oanIhkNbxXnX19RTtkpEL,7KwZNVEaqikRSBSpyhXK2j",
            key=1,
        )

        # slider for probability threshold
        probability_threshold = st.slider("Select probability threshold", 0.5, 1.0, 0.75)

        # streamlit button for getting include recommendations
        if st.button("Include Tracks"):

            # get tracks meta
            track_meta_include = spotify_api.get_tracks_metadata(track_ids_include)

            # hit spotify API
            af_include = spotify_api.get_audio_features(track_ids_include)
            # merge metadata and audio features
            track_df_include = pd.merge(track_meta_include, af_include, on="id")
            # extract artist name for easier reading
            track_df_include["artist"] = track_df_include["artists"].apply(lambda x: x[0]["name"])

            # running inference
            include_df = modeling.include_tracks_to_playlist(
                track_df_include,
                model=st.session_state["model"],
                scaler=st.session_state["scaler"],
                label_encoder=st.session_state["label_encoder"],
                target_playlist_name=target_playlist,
                probability_threshold=0.7,
            )

            # looping trhough track ids
            for track_id in include_df.id.unique():

                # extract data from recommendations dataframe
                track_include = include_df[include_df["id"] == track_id]

                # subheader with track name and artist
                st.subheader(f"{track_include['name'].values[0]} by {track_include['artist'].values[0]}")

                # Embed the audio preview
                preview_url = track_df_include[track_df_include.id == track_id]["preview_url"].values[0]
                if pd.notna(preview_url):  # Ensure the URL is not NaN
                    st.audio(preview_url)
                else:
                    st.write("No audio preview available for this track.")

                # display recommendation
                st.write(
                    "Probability score:", round(track_include["target_playlist_probability"].values[0] * 100, 2), "%"
                )

                # display flag for inclusion
                st.write("Include in playlist:", track_include["include_in_playlist"].values[0])

    else:
        st.warning("Please train a model first.")
