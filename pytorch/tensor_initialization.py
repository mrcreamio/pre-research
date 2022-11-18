import torch
import numpy as np
'''
Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as 
the model’s parameters.

Tensors can be initialized in various ways. Take a look at the following examples:
'''

# Directly from data
data = [[1, 2, 3], [4, 5, 6]]
my_tensor = torch.tensor(data, dtype=torch.float32, device="cpu")
# print(my_tensor)

# From numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# print(x_np)

# from another tensor
x_ones = torch.ones_like(x_np)
x_rand = torch.rand_like(x_ones, dtype=torch.double)
# print(f"random tensor: \n {x_rand} \n")
# print(f"ones tensor: \n {x_ones} \n")

# With random or constant values:
shape = (2,4)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeroes_tenor = torch.zeros(shape)
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"ones tensor: \n {ones_tensor} \n")
# print(f"zeroes tensor \n {zeroes_tenor}")



# Attributes of a Tensor
my_tensor = torch.empty(size = (3, 3))

# print(f"Shape of tensor: {my_tensor.shape}")
# print(f"Datatype of tensor: {my_tensor.dtype}")
# print(f"Device tensor is stored on: {my_tensor.device}")
