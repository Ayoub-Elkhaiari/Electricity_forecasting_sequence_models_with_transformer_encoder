import torch
import matplotlib.pyplot as plt

# def visualize_predictions(model, X_test, y_test, scaler, data):
#     """
#     Function to make predictions using the GRU model and visualize the actual vs predicted values.

#     Parameters:
#     GRU_model (torch.nn.Module): The trained GRU model.
#     X_test (torch.Tensor): The test input features.
#     y_test (torch.Tensor): The actual test values.
#     scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for normalizing the data.
#     data (pandas.DataFrame): The original data containing 'DATE' column.

#     Returns:
#     None
#     """
#     # Set the model to evaluation mode
#     model.eval()
    
#     # Make predictions without calculating gradients
#     with torch.no_grad():
#         predictions = model(X_test)
    
#     # Rescale predictions and actual values back to the original range
#     predicted_values = scaler.inverse_transform(predictions.numpy())
#     actual_values = scaler.inverse_transform(y_test.numpy())
    
#     # Visualize the data before and after prediction
#     plt.figure(figsize=(14, 6))
#     plt.plot(data['DATE'].iloc[-len(y_test):].values, actual_values, label='Actual Data', color='blue')
#     plt.plot(data['DATE'].iloc[-len(y_test):].values, predicted_values, label='Predicted Data', color='red')
#     plt.xlabel('Date')
#     plt.ylabel('IPG2211A2N')
#     plt.title('Time Series Prediction with GRU')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.show()




def visualize_model_predictions(model, X_test, y_test, scaler, data, model_name="Model"):
    """
    Function to make predictions using a given model and visualize the actual vs predicted values.

    Parameters:
    model (torch.nn.Module): The trained PyTorch model (GRU, LSTM, RNN, Transformer, etc.).
    X_test (torch.Tensor): The test input features.
    y_test (torch.Tensor): The actual test values.
    scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for normalizing the data.
    data (pandas.DataFrame): The original data containing 'DATE' column.
    model_name (str): Name of the model for the title of the plot.

    Returns:
    None
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions without calculating gradients
    with torch.no_grad():
        predictions = model(X_test)
    
    # Rescale predictions and actual values back to the original range
    predicted_values = scaler.inverse_transform(predictions.numpy())
    actual_values = scaler.inverse_transform(y_test.numpy())
    
    # Visualize the data before and after prediction
    plt.figure(figsize=(14, 6))
    plt.plot(data['DATE'].iloc[-len(y_test):].values, actual_values, label='Actual Data', color='blue')
    plt.plot(data['DATE'].iloc[-len(y_test):].values, predicted_values, label='Predicted Data', color='red')
    plt.xlabel('Date')
    plt.ylabel('IPG2211A2N')
    plt.title(f'Time Series Prediction with {model_name}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()