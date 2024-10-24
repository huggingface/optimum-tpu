import torch
import torch_xla.core.xla_model as xm

devices = xm.get_xla_supported_devices()
print(f'PyTorch can access {len(devices)} TPU cores')

# Example tensor operations on TPU
dev = xm.xla_device()
print(f"PyTorich device: {dev}")
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)
