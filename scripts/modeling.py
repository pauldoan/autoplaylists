# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# importing DL libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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


# ---------------------------------------------------------------------------- #
#                               Neural net class                               #
# ---------------------------------------------------------------------------- #


# NN model class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=F.relu, dropout=0.1):
        super(NeuralNet, self).__init__()

        # I want my model to be flexible, so I will allow for multiple hidden layers
        layers = []

        # Input layer to first hidden layer, hidden_sizes len should be at least 1
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Dropout(dropout))

        # Adding hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Store the layers in a ModuleList
        self.layers = nn.ModuleList(layers)

        # store the activation function
        self.activation = activation

    # forward pass
    def forward(self, x):
        # Pass input through each layer in the network
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)

            # we add the activation if layer is not dropout
            if isinstance(self.layers[i], nn.Linear):
                x = self.activation(x)

        # no activation for the last layer
        x = self.layers[-1](x)

        return x


# ---------------------------------------------------------------------------- #
#                               Training function                              #
# ---------------------------------------------------------------------------- #


# little training function
def train_model(model, train_loader, eval_loader, optimizer, device="cpu", epochs=10):

    # move model to device
    model.to(device)

    # history
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    # setup loss function for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # training loop
    print("Training Start")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0
        for x, y in train_loader:
            # move data to device
            x = x.to(device)
            y = y.to(device)

            # forward pass
            outputs = model(x)
            cur_train_loss = criterion(outputs, y)
            cur_train_acc = (outputs.argmax(dim=1) == y).float().mean().item()

            # backpropagation
            optimizer.zero_grad()
            cur_train_loss.backward()

            # update weights
            optimizer.step()

            # loss and acc
            train_loss += cur_train_loss
            train_acc += cur_train_acc

        # valid start
        model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(device)
                y = y.to(device)

                # predict
                outputs = model(x)
                cur_valid_loss = criterion(outputs, y)
                cur_valid_acc = (outputs.argmax(dim=1) == y).float().mean().item()

                # loss and acc
                valid_loss += cur_valid_loss
                valid_acc += cur_valid_acc

        # epoch output
        train_loss = (train_loss / len(train_loader)).item()
        train_acc = train_acc / len(train_loader)
        val_loss = (valid_loss / len(eval_loader)).item()
        val_acc = valid_acc / len(eval_loader)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(val_loss)
        history["valid_acc"].append(val_acc)
        print(
            f"Epoch:{epoch + 1} / {epochs}, train loss:{train_loss:.4f} train_acc:{train_acc:.4f}, valid loss:{val_loss:.4f} valid acc:{val_acc:.5f}"
        )

    return model, history


# ---------------------------------------------------------------------------- #
#                      Full building and training function                     #
# ---------------------------------------------------------------------------- #


def build_train(df, epochs=10, hidden_sizes=[8, 8], lr=0.01, activation=F.relu, dropout=0.1):

    # extracting independent variables
    X = df[features]

    # extracting dependent variable
    y = df[["playlist_name"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

    # Encoding the dependent variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.squeeze())
    y_test_encoded = label_encoder.transform(y_test.squeeze())

    # feature selection - we could automate this process but i kept it static for now
    # dropping the least important features
    X_train_scaled.drop(columns=["loudness", "acousticness"], inplace=True)
    X_test_scaled.drop(columns=["loudness", "acousticness"], inplace=True)
    X_train_scaled.drop(columns=["time_signature", "key", "mode"], inplace=True)
    X_test_scaled.drop(columns=["time_signature", "key", "mode"], inplace=True)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    # creating clean torch datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # creating dataloaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # initializing the model
    input_size = X_train_scaled.shape[1]
    output_size = len(label_encoder.classes_)
    model = NeuralNet(
        input_size,
        hidden_sizes,
        output_size,
        activation=activation,
        dropout=dropout,
    )

    # training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, history = train_model(
        model,
        train_loader=train_loader,
        eval_loader=test_loader,
        optimizer=optimizer,
        device="cpu",
        epochs=epochs,
    )

    # evaluation on the test set
    model.eval()
    with torch.no_grad():
        y_pred = model(test_loader.dataset.tensors[0])

    # converting logits to class
    y_pred_class = y_pred.argmax(dim=1).numpy()

    # printing a little classification report
    class_report = classification_report(
        y_test_encoded, y_pred_class, target_names=label_encoder.classes_, output_dict=True
    )

    # Convert the dictionary to a DataFrame
    class_report = pd.DataFrame(class_report).transpose()

    return model, label_encoder, scaler, history, class_report


# ---------------------------------------------------------------------------- #
#                               Plotting function                             #
# ---------------------------------------------------------------------------- #


def plot_training_history(history):

    # plotting training history
    train_loss_hist = history["train_loss"]
    val_loss_hist = history["valid_loss"]
    train_acc_hist = history["train_acc"]
    val_acc_hist = history["valid_acc"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(train_loss_hist, ax=axes[0], label="Training loss")
    sns.lineplot(val_loss_hist, ax=axes[0], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    sns.lineplot(train_acc_hist, ax=axes[1], label="Training Acc")
    sns.lineplot(val_acc_hist, ax=axes[1], label="Validation Acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    return fig, axes


# ---------------------------------------------------------------------------- #
#                              Inference function                              #
# ---------------------------------------------------------------------------- #


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
    df["predicted_probability"] = y_prob.max(dim=1).values.numpy()
    df["all_probabilities"] = genre_probs

    return df


# ---------------------------------------------------------------------------- #
#                           include track to playlist                          #
# ---------------------------------------------------------------------------- #


def include_tracks_to_playlist(df, model, scaler, label_encoder, target_playlist_name, probability_threshold=0.75):

    # run inference
    df = model_inference(df, model, scaler, label_encoder)

    # filter data
    target_playlist_df = df[["id", "name", "artist", "all_probabilities"]].copy()
    target_playlist_df["target_playlist_name"] = target_playlist_name
    target_playlist_df["target_playlist_probability"] = target_playlist_df["all_probabilities"].apply(
        lambda x: x[target_playlist_name]
    )

    # flag based on probability threshold
    target_playlist_df["include_in_playlist"] = (
        target_playlist_df["target_playlist_probability"] > probability_threshold
    )
    return target_playlist_df
