import torch

# this will return error because not float
# x = torch.tensor([[1,0],[-1,1]], requires_grad=True)

x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
print(x)
print(z)

z.backward()

print(x.grad)
