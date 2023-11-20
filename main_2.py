import torch

randint = torch.randint(-100, 100, (6,))
randint

tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor

zeros = torch.zeros(2, 3)
zeros

ones = torch.ones(3, 4)
ones

input = torch.empty(2, 3)
input

arange = torch.arange(5)
arange

linspace = torch.linspace(3, 10, steps = 5)
linspace

