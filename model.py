import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class NeuralCA(nn.Module):
    def __init__(self, square_size, embedding_size, num_heads, num_encoders, num_classes):
        super(NeuralCA, self).__init__()
        self.square_size = square_size
        self.embedding_size = embedding_size
        
        # Embedding layer for palette indices
        self.embedding = nn.Embedding(num_classes, embedding_size)
        
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoders)
        
        # Fully connected layer to predict the middle pixel
        self.fc = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # Convert palette indices to embeddings
        x = x.permute(1, 0, 2)  # Shape needed for Transformer: (seq_len, batch_size, embedding_size)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over the sequence
        x = self.fc(x)
        return x

def get_model(config):
    return NeuralCA(
        square_size=config["square_size"],
        embedding_size=config["embedding_size"],
        num_heads=config["num_heads"],
        num_encoders=config["num_encoders"],
        num_classes=config["num_classes"],
    )
