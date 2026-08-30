import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	a = np.array(a)
	b = np.array(b)
	if a.ndim == 1: 
		print("Must have dim greater than 1")
		return -1
	
	a_rows, a_cols = a.shape

	if a_cols != len(b):
		return -1
	
	return np.matmul(a,b)