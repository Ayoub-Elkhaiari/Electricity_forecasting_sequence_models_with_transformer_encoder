from models.LSTM_model import LSTMModel
from models.LSTM_with_Attention import LSTMWithAttentionModel
from models.RNN_model import RNNModel
from models.GRU_model import GRUModel
from models.Transformer_encoder_model import TransformerModel

# from visualize import visualize_model_predictions
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch 
from torch import nn
import math
from sklearn.metrics import mean_squared_error






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

def train_model(model, X_train, y_train, criterion, optimizer, num_epochs=300, print_interval=10):
    """
    Train a PyTorch model.

    Returns:
    Trained model.
    """
    model.train()
    for epoch in range(num_epochs):
        output = model(X_train)
        loss = criterion(output, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % print_interval == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the model and return RMSE.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    predicted_values = scaler.inverse_transform(predictions.numpy())
    actual_values = scaler.inverse_transform(y_test.numpy())
    rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
    return rmse, predicted_values, actual_values

def visualize_model_predictions(models, X_test, y_test, scaler, data):
    """
    Evaluate and visualize multiple models, ranking them from best to worst.
    
    Parameters:
    models (list): List of tuples [(model_instance, model_name), ...]
    """
    results = []
    
    # Evaluate each model
    for model, model_name in models:
        print(f"\nEvaluating {model_name}...")
        rmse, predicted_values, actual_values = evaluate_model(model, X_test, y_test, scaler)
        results.append((model_name, rmse, predicted_values, actual_values))
    
    # Sort models by performance (RMSE)
    results.sort(key=lambda x: x[1])
    
    # Visualize predictions
    for i, (model_name, rmse, predicted_values, actual_values) in enumerate(results):
        print(f"{i+1}. {model_name}: RMSE = {rmse:.4f}")
        plt.figure(figsize=(14, 6))
        plt.plot(data['DATE'].iloc[-len(y_test):].values, actual_values, label='Actual Data', color='blue')
        plt.plot(data['DATE'].iloc[-len(y_test):].values, predicted_values, label='Predicted Data', color='red')
        plt.xlabel('Date')
        plt.ylabel('IPG2211A2N')
        plt.title(f'Time Series Prediction with {model_name} (RMSE: {rmse:.4f})')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

# Instantiate models
model_rnn = RNNModel()
model_lstm = LSTMModel()
model_lstm_attention = LSTMWithAttentionModel()
model_gru = GRUModel()
model_transformer = TransformerModel(input_size=X_train.shape[2], hidden_size=64, output_size=1)

# Define loss and optimizer
criterion = nn.MSELoss()
learning_rate = 0.001

# Train all models
trained_models = []

# RNN
optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=learning_rate)
trained_rnn = train_model(model_rnn, X_train, y_train, criterion, optimizer_rnn)
trained_models.append((trained_rnn, "RNN"))

# LSTM
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)
trained_lstm = train_model(model_lstm, X_train, y_train, criterion, optimizer_lstm)
trained_models.append((trained_lstm, "LSTM"))

# LSTM with Attention
optimizer_lstm_attention = torch.optim.Adam(model_lstm_attention.parameters(), lr=learning_rate)
trained_lstm_attention = train_model(model_lstm_attention, X_train, y_train, criterion, optimizer_lstm_attention)
trained_models.append((trained_lstm_attention, "LSTM with Attention"))

# GRU
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=learning_rate)
trained_gru = train_model(model_gru, X_train, y_train, criterion, optimizer_gru)
trained_models.append((trained_gru, "GRU"))

# Transformer
optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=learning_rate)
trained_transformer = train_model(model_transformer, X_train, y_train, criterion, optimizer_transformer)
trained_models.append((trained_transformer, "Transformer"))

# Visualize results from best to worst
visualize_model_predictions(trained_models, X_test, y_test, scaler, data)
