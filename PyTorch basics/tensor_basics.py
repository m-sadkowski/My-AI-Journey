"""
-----------------------------------------------------------------------------------------
source of knowledge: https://www.youtube.com/watch?v=d2dS8kLQHco&ab_channel=RobertSikora
-----------------------------------------------------------------------------------------
"""

import torch

# Check if CUDA (GPU support) is available, and set the device accordingly
if torch.cuda.is_available():
    device = 'cuda' 
else:
    device = 'cpu'  

tensor_1 = torch.tensor([[2, 4, 5],
                         [1, 1, 1],
                         [3, 1, 0]])

tensor_2 = torch.tensor([[2, 4, 5],
                         [1, 1, 1],
                         [3, 1, 0]])

tensor_1_move_to_gpu_or_cpu = tensor_1.to(device=device)

tensor_2_move_to_gpu_or_cpu = tensor_2.to(device=device)

tensor_sum = tensor_1_move_to_gpu_or_cpu + tensor_2_move_to_gpu_or_cpu

# Print the number of dimensions (ndim) of the resulting tensor
print(tensor_sum.ndim)  # Output: 2 (since it's a 2D tensor)

# Print the shape of the resulting tensor
print(tensor_sum.shape)  # Output: torch.Size([3, 3]) (3 rows and 3 columns)