import ctypes
import numpy as np

# Load the shared library with an absolute path
lib = ctypes.CDLL(r'C:\Users\oahre\git_Repos\GPU_Playground\01_dot_product\src\libdotproduct_cuda.dll')

# Define argument types for the function
lib.dot_product.argtypes = [ctypes.POINTER(ctypes.c_float),
                             ctypes.POINTER(ctypes.c_float),
                             ctypes.POINTER(ctypes.c_float),
                             ctypes.c_int]

def dot_prod(arr1: np.ndarray, arr2: np.ndarray) -> float:
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape.")

    size = arr1.size
    
    # Create empty result array on host (CPU)
    result = np.zeros(1, dtype=np.float32)

    # Convert NumPy arrays to ctypes pointers
    a_ptr = arr1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = arr2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the C++ function in the shared library
    lib.dot_product(a_ptr,
                    b_ptr,
                    result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    size)

    return result[0]

# Example usage:
a = np.array([1.0, 2.0], dtype=np.float32)
b = np.array([2.0, 5.0], dtype=np.float32)

result = dot_prod(a,b)
print(f"The dot product is: {result}")