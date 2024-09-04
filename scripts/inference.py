import pandas as pd
import torch

# Selecting the relevant audio features
features = [
    "duration_ms",
    "popularity",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


def model_inference(df, model, scaler, label_encoder):
    # Selecting the relevant features
    X = df[features]

    # Scaling the features
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # dropping features from feature selection
    # dropping loudness and acousticness
    X_scaled.drop(columns=["loudness", "acousticness", "time_signature", "key", "mode"], inplace=True)

    # converting the input to tensor
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)

    # inference
    with torch.no_grad():
        y_logit = model(X_tensor)
        y_prob = torch.softmax(y_logit, dim=1)
        y_pred = torch.argmax(y_logit, dim=1)
        y_pred = label_encoder.inverse_transform(y_pred.numpy())

    # creating dict of genre and predicted probability for each entry in the list
    genre_probs = []
    for i in range(len(y_prob)):
        genre_prob = dict(zip(label_encoder.classes_, y_prob[i].numpy().tolist()))
        genre_probs.append(genre_prob)

    # saving data in orginial dataframe
    df["predicted_label"] = y_pred
    df["predicted_probabilty"] = y_prob.max(dim=1).values.numpy()
    df["all_probabilities"] = genre_probs

    return df
