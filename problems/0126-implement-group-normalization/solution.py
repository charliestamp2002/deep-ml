import torch

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero
    
    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """
    # reshape tensor to go from (B, C, H, W) to (B, G, C//G, H, W)
    B, C, H, W = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    G = num_groups
    X = X.reshape(B, G, C // G, H, W)
    print(X.shape)
    mean = X.mean(dim=(-3,-2,-1), keepdim=True)
    print(mean.shape)
    variance = X.var(dim=(-3,-2,-1), correction=0, keepdim=True)
    print(variance.shape)
    X_norm = (X - mean) / torch.sqrt(variance + epsilon)
    X_norm = X_norm.reshape(B, C, H, W)
    X_norm = gamma*X_norm + beta
    return X_norm



