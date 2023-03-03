import torch
import torchaudio

Lin = 4800
DesiredLout = 257*188
stride = 10
padding= 0

# kernel_size = 21
output_padding = 9

# output padding must be smaller than either stride or dilation

enc_kernel_size = 317
# output_padding=(enc_kernel_size // 2) - 1
kernel_size=enc_kernel_size
# stride=enc_kernel_size // 2
# padding=enc_kernel_size // 2

Lout = ((Lin - 1)*stride) - (2*padding) + (kernel_size-1) + output_padding + 1

print(Lout)
print(DesiredLout)
