import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        """
        Initialize Additive Attention.
        
        Args:
        - query_dim: Dimension of the query vector (decoder hidden state).
        - key_dim: Dimension of the key vector (encoder hidden state).
        - hidden_dim: Dimension of the hidden layer in the feedforward network.
        """
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(query_dim, hidden_dim, bias=False)  # Linear layer for query
        self.W2 = nn.Linear(key_dim, hidden_dim, bias=False)   # Linear layer for key
        self.V = nn.Linear(hidden_dim, 1, bias=False)          # Linear layer for scoring

    def forward(self, query, keys):
        """
        Forward pass for additive attention.

        Args:
        - query: Tensor of shape (batch_size, query_dim), the query vector.
        - keys: Tensor of shape (batch_size, seq_len, key_dim), the key vectors.

        Returns:
        - context: Tensor of shape (batch_size, key_dim), the context vector.
        - attention_weights: Tensor of shape (batch_size, seq_len), attention weights.
        """
        # Expand query to match the keys' sequence length
        print(query)
        query_expanded = query.unsqueeze(1)  # (batch_size, 1, query_dim)
        print(self.W1(query_expanded).shape)
        # Compute the attention scores
        score = self.V(torch.tanh(self.W1(query_expanded) + self.W2(keys)))  # (batch_size, seq_len, 1)
        score = score.squeeze(-1)  # (batch_size, seq_len)
        
        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(score, dim=1)  # (batch_size, seq_len)
        
        # Compute the context vector as a weighted sum of the keys
        context = torch.bmm(attention_weights.unsqueeze(1), keys)  # (batch_size, 1, key_dim)
        context = context.squeeze(1)  # (batch_size, key_dim)
        
        return context, attention_weights

# Example usage
batch_size = 2
seq_len = 5
query_dim = 4
key_dim = 4
hidden_dim = 8

# Sample query and keys
query = torch.rand(batch_size, query_dim)  # (batch_size, query_dim)
keys = torch.rand(batch_size, seq_len, key_dim)  # (batch_size, seq_len, key_dim)

# Initialize and compute attention
attention = AdditiveAttention(query_dim, key_dim, hidden_dim)
context, attention_weights = attention(query, keys)

# print("Context Vector:\n", context.shape)
# print("Attention Weights:\n", attention_weights)
