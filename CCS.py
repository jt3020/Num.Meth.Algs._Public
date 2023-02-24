import math 
import numpy as np 

def ccs_matrix(matrix):
    rows, cols = matrix.shape
    values = []
    col_indices = []
    row_pointers = [0]
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 0:
                values.append(matrix[i, j])
                col_indices.append(j)
        row_pointers.append(len(values))
    
    return values, col_indices, row_pointers


def to_crs(matrix):
  # Get the non-zero elements and their indices
  nonzero_elements = matrix[matrix != 0]
  nonzero_indices = np.argwhere(matrix != 0)

  # Create the value array by flattening the non-zero elements
  value_array = nonzero_elements.flatten()

  # Create the index array by extracting the column indices of the non-zero elements
  index_array = nonzero_indices[:, 1]

  # Create the pointer array
  num_rows = matrix.shape[0]
  pointer_array = np.empty(num_rows + 1, dtype=int)
  
  # Set the first element of the pointer array to zero
  pointer_array[0] = 0

  # Fill the pointer array
  current_index = 0
  for i in range(len(nonzero_elements)):
    if (nonzero_indices[i,0] > current_index):
        current_index += 1
        pointer_array[current_index] = i
    
  # Set the last element of the pointer array to the length of the value array
  pointer_array[-1] = len(value_array)

  return value_array, index_array, pointer_array


def crs_matvec(value_array, index_array, pointer_array, vec):
  # Initialize the output vector
  result = np.zeros(len(pointer_array)-1)

  # Iterate over the rows of the matrix
  for i in range(len(pointer_array) - 1):
    # Extract the non-zero elements and their indices for row i
    start = pointer_array[i]
    end = pointer_array[i + 1]
    row_values = value_array[start:end]
    row_indices = index_array[start:end]

    # Multiply the elements by the corresponding elements of the input vector, and sum the products
    result[i] = np.sum(row_values * vec[row_indices])

  return result

# Test the function with a simple matrix
matrix = np.array([[1, 0, 3, 0], [0, 0, 0, 4], [5, 0, 0, 0]])
matrix = np.array([[1, 0, 3, 0],
 [0, 4, 0, 0],
 [0, 0, 0, 5]])
value_array, index_array, pointer_array = to_crs(matrix)



# print('Input Matrix:')
# print(matrix)
# print('CRS Value Array:')
# print(value_array)
# print('CRS Index Array:')
# print(index_array)
# print('CRS Pointer Array:')
# print(pointer_array)
# print('')

value_array, index_array, pointer_array = ccs_matrix(matrix)

print('CCS Value Array:')
print(value_array)
print('CCS Col Array:')
print(index_array)
print('CCS Row Array:')
print(pointer_array)
print('')

# vec = np.array([1,2,3,4])
# print('Input Vector:')
# print(vec)
# print('Output:')
# print(crs_matvec(value_array, index_array, pointer_array, vec))