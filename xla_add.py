import torch
import torch_xla
import torch_xla.core.xla_model as xm

def simple_xla_calc():
    # Get XLA device
    device = xm.xla_device()
    
    # Create tensors on XLA device
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform matrix multiplication
    result = torch.matmul(x, y)
    
    # Force computation and synchronize
    result_cpu = result.cpu()
    
    return result_cpu

# Run calculation
result = simple_xla_calc()
print(f"Result shape: {result.shape}")