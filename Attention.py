import numpy as np

def attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        query (np.ndarray): Query matrix of shape (batch_size, num_heads, seq_len, head_dim).
        key (np.ndarray): Key matrix of shape (batch_size, num_heads, seq_len, head_dim).
        value (np.ndarray): Value matrix of shape (batch_size, num_heads, seq_len, head_dim).
        mask (np.ndarray): Optional mask of shape (batch_size, 1, seq_len, seq_len).
        
    Returns:
        np.ndarray: Attention output of shape (batch_size, num_heads, seq_len, head_dim).
        np.ndarray: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
    """
    # Step 1: Compute the dot products between query and key
    scores = np.matmul(query, key.transpose(0, 1, 3, 2))
    
    # Scale scores by the square root of head dimension
    d_k = query.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    
    # Step 2: Apply mask (optional)
    if mask is not None:
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)
    
    # Step 3: Compute softmax over the last axis (attention weights)
    attention_weights = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Step 4: Weighted sum of the value vectors
    attention_output = np.matmul(attention_weights, value)
    
    return attention_output, attention_weights


# Example Usage
batch_size = 2
num_heads = 4
seq_len = 5
head_dim = 8

# Randomly initialize query, key, and value
query = np.random.rand(batch_size, num_heads, seq_len, head_dim)
key = np.random.rand(batch_size, num_heads, seq_len, head_dim)
value = np.random.rand(batch_size, num_heads, seq_len, head_dim)

# Optional mask
mask = np.random.choice([0, 1], size=(batch_size, 1, seq_len, seq_len))

# Call the attention function
output, weights = attention(query, key, value, mask)

print("Attention Output:\n", output.shape)
print("\nAttention Weights:\n", weights.shape)
