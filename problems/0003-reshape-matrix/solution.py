import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method

	flat = []
	for row in a:
		for val in row: 
			flat.append(val)

	rows, cols = new_shape

	if len(flat) != rows * cols: 
		return []
	
	reshaped_matrix = []

	for i in range(rows): 
		new_row = []
		for j in range(cols):
			new_row.append(flat[(i * cols) + j])

		reshaped_matrix.append(new_row)

	return reshaped_matrix