import torch 
from matplotlib import pylab as plt 
from sklearn.metrics import mean_squared_error
from torch import nn 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# importing our models

from models.LSTM_model import LSTMModel
from models.LSTM_with_Attention import LSTMWithAttentionModel
from models.RNN_model import RNNModel
from models.GRU_model import GRUModel
from models.Transformer_encoder_model import TransformerModel

# Function to train and evaluate multiple models with visualization
def train_and_evaluate_all_models(models, X_train, y_train, X_test, y_test, criterion, optimizers, num_epochs=100):
    model_losses = {}
    
    # Train and evaluate each model
    for i, model in enumerate(models):
        optimizer = optimizers[i]
        
        # Training the model
        model.train()
        for epoch in range(num_epochs):
            output = model(X_train)
            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluating the model
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
        loss = mean_squared_error(y_test.numpy(), predictions.numpy())
        
        # Store the loss for this model
        model_losses[type(model).__name__] = loss
    
    # Sort the models by loss (from best to worst)
    sorted_models = sorted(model_losses.items(), key=lambda x: x[1])

    # Visualize the results
    model_names = [model[0] for model in sorted_models]
    losses = [model[1] for model in sorted_models]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(model_names, losses, color='skyblue')
    plt.xlabel('Mean Squared Error')
    plt.title('Model Performance Comparison')
    plt.gca().invert_yaxis()  # Highest performance at the top
    plt.show()
    
    # Return the best model (with the smallest MSE)
    best_model_name, best_loss = sorted_models[0]
    # print(f"Best Model: {best_model_name} with MSE loss: {best_loss}")
    
    # Optionally, return the best model object if you need it
    best_model = [model for model in models if type(model).__name__ == best_model_name][0]
    return best_model, best_loss




# our time serie data 

# Load your data
data = pd.read_csv('Electric_Production.csv')

# Convert 'DATE' to datetime
data['DATE'] = pd.to_datetime(data['DATE'])

# Extract the target variable (e.g., 'IPG2211A2N')
target = data['IPG2211A2N'].values

# Normalize the target variable
scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaled = scaler.fit_transform(target.reshape(-1, 1))

# Create sequences for LSTM (sequence_length determines how many previous timesteps are used to predict the future)
sequence_length = 10

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Create sequences of data
sequences, labels = create_sequences(target_scaled, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, shuffle=False)

# Convert the data into torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Instantiate models
model_lstm = LSTMModel()
model_lstm_attention = LSTMWithAttentionModel()
model_rnn = RNNModel()
model_gru = GRUModel()
model_transformer = TransformerModel(input_size=X_train.shape[2], hidden_size=64, output_size=1)

# Create list of models
models = [model_lstm, model_lstm_attention, model_rnn, model_gru, model_transformer]

# Define loss function
criterion = nn.MSELoss()

# Define optimizers for each model
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
optimizer_lstm_attention = torch.optim.Adam(model_lstm_attention.parameters(), lr=0.001)
optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=0.001)
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=0.001)
optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=0.001)

# Create list of optimizers
optimizers = [optimizer_lstm, optimizer_lstm_attention, optimizer_rnn, optimizer_gru, optimizer_transformer]

# Train and evaluate all models, also visualizing the results
best_model, best_loss = train_and_evaluate_all_models(models, X_train, y_train, X_test, y_test, criterion, optimizers)

print(f"Best Model: {type(best_model).__name__} with MSE loss: {best_loss}")

