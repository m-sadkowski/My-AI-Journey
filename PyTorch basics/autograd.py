import torch

x = torch.tensor(2.0, requires_grad = True)
y = torch.tensor(5.0, requires_grad = True)

z = 2 * x + y

print(f"Value of function z: {z.item()}")

# Gradient (dz/dx and dz/dy)
z.backward()

print(f"dz/dx: {x.grad}")
print(f"dz/dy: {y.grad}")