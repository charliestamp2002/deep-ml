import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
	np.random.seed(seed)
	def dropout(x, p):
		mask = np.random.binomial(1, 1-p, size=x.shape)
		return x * mask / (1-p)

	W1 = np.array(W1)
	x = np.array(x)
	x = np.expand_dims(x, axis=1)
	b1 = np.array(b1)
	b1 = np.expand_dims(b1, axis=1)

	l1 = W1 @ x + b1
	l1 = np.maximum(0,l1)
	
	W2 = np.array(W2)
	y = W2 @ l1 + np.expand_dims(b2, axis=1)
	
	y = dropout(y, dropout_p)
	return np.squeeze(y + x)



	
	