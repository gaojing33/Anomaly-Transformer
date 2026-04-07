import torch
import torch_directml

dml = torch_directml.device()
print("device:", dml)

x = torch.randn(2, 3).to(dml)
y = torch.randn(2, 3).to(dml)
z = x + y
print(z)