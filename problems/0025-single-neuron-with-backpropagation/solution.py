import torch


def train_neuron(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    full-batch gradient descent with MSE loss.

    Args:
        features: Input tensor of shape (n_samples, n_features)
        labels: Binary labels of shape (n_samples,)
        initial_weights: Initial weights of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values),
        rounded to 4 decimal places.
    """

    # Initialise parameters
    updated_weights = initial_weights.clone().unsqueeze(0)

    updated_bias = torch.tensor(
        initial_bias,
        dtype=features.dtype,
        device=features.device
    )

    # Reshape labels to (1, n_samples)
    labels = labels.unsqueeze(0)

    mse_values = torch.zeros(epochs, dtype=features.dtype)

    def _forward_pass(
        weights: torch.Tensor,
        feats: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:

        predictions = torch.zeros(
            (1, len(feats)),
            dtype=feats.dtype,
            device=feats.device
        )

        for i in range(len(feats)):
            x = feats[i].unsqueeze(1)  # (n_features, 1)
            z = weights @ x + bias     # (1, 1)
            predictions[0, i] = torch.sigmoid(z).squeeze()

        return predictions

    def _calculate_mse(
        predictions: torch.Tensor,
        true_labels: torch.Tensor
    ) -> torch.Tensor:

        return torch.mean((predictions - true_labels) ** 2)

    def _calculate_gradients(
        predictions: torch.Tensor,
        true_labels: torch.Tensor,
        feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Error signal from the chain rule
        delta = (
            2
            * (predictions - true_labels)
            * predictions
            * (1 - predictions)
        )  # (1, n_samples)

        # Average gradients across the full batch
        n_samples = len(feats)

        weight_grads = (delta @ feats) / n_samples
        # (1, n_samples) @ (n_samples, n_features)
        # -> (1, n_features)

        bias_grad = torch.sum(delta) / n_samples

        return weight_grads, bias_grad

    # Full-batch gradient descent
    for epoch in range(epochs):

        # 1. Forward pass using current parameters
        predictions = _forward_pass(
            updated_weights,
            features,
            updated_bias
        )

        # 2. Record MSE before updating parameters
        mse = _calculate_mse(predictions, labels)
        mse_values[epoch] = mse

        # 3. Calculate gradients using the same predictions
        weight_grads, bias_grad = _calculate_gradients(
            predictions,
            labels,
            features
        )

        # 4. Update parameters once for the entire batch
        updated_weights = (
            updated_weights - learning_rate * weight_grads
        )

        updated_bias = updated_bias - learning_rate * bias_grad

    # Round only the final returned values
    final_weights = [
        round(w, 4)
        for w in updated_weights.squeeze(0).tolist()
    ]

    final_bias = round(updated_bias.item(), 4)

    final_mse_values = [
        round(mse, 4)
        for mse in mse_values.tolist()
    ]

    return final_weights, final_bias, final_mse_values