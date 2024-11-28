import torch
import torch_xla.core.xla_model as xm

def simple_xla_calc():
    device = xm.xla_device()
    
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    result = torch.matmul(x, y)
    
    result_cpu = result.cpu()
    
    return result_cpu

# Run calculation
result = simple_xla_calc()
print(f"Result shape: {result.shape}")