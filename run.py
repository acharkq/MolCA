import torch
print(torch.cuda.is_available())
a = torch.tensor([1, 2, 3, 4]).to('cuda:0')