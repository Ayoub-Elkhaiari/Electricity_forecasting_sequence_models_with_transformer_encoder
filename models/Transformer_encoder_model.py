import torch.nn as nn



class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        
        # Encoder layer (from nn.TransformerEncoder)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Transform input sequence to embeddings
        x = self.embedding(x)
        
        # Apply Transformer Encoder
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, feature_size)
        x = self.transformer_encoder(x)
        
        # Use the output of the last time step for prediction (seq_len -> batch_size -> output)
        x = x[-1, :, :]  # Taking the last time step
        x = self.fc_out(x)
        return x