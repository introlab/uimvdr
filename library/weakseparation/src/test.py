import torch

dummy = torch.Tensor([[0,1,2,3,4,5,6,7,8,9,10,11]])

max_size = 10

new_dummy = dummy[..., :max_size-dummy.shape[1]]

print(new_dummy.shape)
