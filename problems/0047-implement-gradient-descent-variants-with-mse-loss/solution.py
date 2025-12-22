import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):

    def compute_gradient(X, y, weights): 
        N = len(X)
        preds = X @ weights
        error = preds - y
        gradient = (2/N) * (error @ X)
        return gradient 

    N = len(y)
    start = 0

    for _ in range(n_iterations): 

        if method == 'batch': 
            batch_size = N
        elif method == "stochastic": 
            batch_size = 1
        else: 
            batch_size = batch_size
        
        indices = np.arange(N)

        for start in range(0, N, batch_size):

            end = min(start + batch_size, N)
            batch_idx = indices[start: end]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            grad = compute_gradient(X_batch, y_batch, weights)
            
            weights = weights - learning_rate * grad

    return weights

	


