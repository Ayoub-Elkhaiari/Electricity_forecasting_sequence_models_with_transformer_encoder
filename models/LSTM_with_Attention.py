import torch
import torch.nn as nn

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weight = nn.Parameter(torch.randn(hidden_size, 1))
        
    def forward(self, lstm_outputs):
        # Compute attention scores
        attn_scores = torch.matmul(lstm_outputs, self.attention_weight)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Softmax over sequence length
        context_vector = torch.sum(attn_scores * lstm_outputs, dim=1)
        return context_vector


class LSTMWithAttentionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMWithAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # First LSTM layer
        out, _ = self.lstm1(x, (h0, c0))
        
        # Apply attention after the first LSTM layer
        context_vector = self.attention(out)
        
        # Second LSTM layer (using the context vector as input)
        out, _ = self.lstm2(context_vector.unsqueeze(1).repeat(1, x.size(1), 1), (h0, c0))
        
        # Output layer
        out = out[:, -1, :]
        out = self.fc(out)
        return out