# Time Series Prediction with Neural Networks (LSTM,LSTM with attention, GRU, RNN, and Transformer models.)

A comprehensive PyTorch implementation for time series prediction using various neural network architectures, including LSTM, GRU, RNN, and Transformer models. This project specifically focuses on electric production forecasting.

## 🌟 Features

- Multiple neural network architectures:
  - Simple RNN
  - LSTM (Long Short-Term Memory)
  - LSTM with Attention Mechanism
  - GRU (Gated Recurrent Unit)
  - Transformer Encoder
- Automated model comparison and visualization
- Data preprocessing utilities
- Performance metrics visualization
- Time series sequence creation

## 📋 Prerequisites

```bash
python >= 3.6
torch
pandas
numpy
matplotlib
scikit-learn
```

## 🎯 Project Structure

```
├── models/
│   ├── LSTM_model.py
│   ├── LSTM_with_Attention.py
│   ├── RNN_model.py
│   ├── GRU_model.py
│   └── Transformer_encoder_model.py
├── main_with_all_models.py
├── main_with_each_model.py
└── Electric_Production.csv
```

## 💻 Usage

1. Train and compare all models:

```python
from models.LSTM_model import LSTMModel
from models.LSTM_with_Attention import LSTMWithAttentionModel
from models.RNN_model import RNNModel
from models.GRU_model import GRUModel
from models.Transformer_encoder_model import TransformerModel

# Initialize models
model_lstm = LSTMModel()
model_lstm_attention = LSTMWithAttentionModel()
model_rnn = RNNModel()
model_gru = GRUModel()
model_transformer = TransformerModel(input_size=1, hidden_size=64, output_size=1)

# Train and evaluate
best_model, best_loss = train_and_evaluate_all_models(
    models=[model_lstm, model_lstm_attention, model_rnn, model_gru, model_transformer],
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    criterion=criterion,
    optimizers=optimizers
)
```

2. Train individual models:

```python
# Example with LSTM
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trained_model = train_model(model, X_train, y_train, criterion, optimizer)
```

## 🔧 Model Architectures

### LSTM Model
- Basic LSTM architecture
- 2 LSTM layers with 50 hidden units
- Fully connected output layer

### LSTM with Attention
- Dual LSTM layers
- Custom attention mechanism
- Context vector generation
- Enhanced feature focusing

### GRU Model
- Gated Recurrent Unit architecture
- 2 GRU layers
- Simplified gating mechanism

### RNN Model
- Simple RNN architecture
- 2 RNN layers
- Basic sequential learning

### Transformer Model
- Transformer encoder architecture
- Multi-head attention
- Positional encoding
- Linear embedding layer

## 📊 Data Preprocessing

The project includes utilities for:
- Time series sequence creation
- Data normalization using MinMaxScaler
- Train-test splitting
- Tensor conversion

```python
# Create sequences
sequence_length = 10
sequences, labels = create_sequences(data, sequence_length)

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(
    sequences, 
    labels, 
    test_size=0.2, 
    shuffle=False
)
```

## 📈 Visualization

The project provides visualization utilities for:
- Model performance comparison
- Prediction vs actual values
- Time series forecasting results

## Results: 

- in `main_with_each_model.py` :
  
![Screenshot 2024-11-14 081510](https://github.com/user-attachments/assets/abca57e8-0ba2-40ff-bc1e-2c1d35b36e43)

![Screenshot 2024-11-14 081523](https://github.com/user-attachments/assets/f9e9762e-6ff8-4702-abc4-8f352ada36df)

![Screenshot 2024-11-14 081536](https://github.com/user-attachments/assets/610d503c-8cb2-4654-bce9-ee43a106f09b)

![Screenshot 2024-11-14 081549](https://github.com/user-attachments/assets/9706c0b8-a037-4213-bf72-d85e33416855)

![Screenshot 2024-11-14 081627](https://github.com/user-attachments/assets/29ecb465-aa81-424c-9ba2-d789e1db9d8c)

- in `main_with_all_models.py` :

![Screenshot 2024-11-14 081904](https://github.com/user-attachments/assets/8a2eeac5-542e-472d-90d1-816a0f1f79f3)


We observe that the RNN model outperforms the other models in this use case. The primary reason is the small size of the dataset. Models like LSTM, GRU, and Transformers are designed to capture long-range dependencies in sequences, which is advantageous for larger datasets with complex temporal patterns. However, given the limited data, the simpler RNN architecture is sufficient and more effective, as it doesn't require the complexity needed to learn long-range dependencies, thus performing better in this scenario.


## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📝 License

