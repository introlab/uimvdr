import torch

mix = torch.tensor([[1,1],[2, 2],[3,3]])

masks = torch.tensor([[[1,1],[1,1],[1,1]],
                     [[2,2],[2,2],[2,2]],
                     [[3,3],[3,3],[3,3]]])

# pred = torch.einsum("ij,ik->ij", masks, mix)

pred = mix * masks

print(pred[:,1,:])
