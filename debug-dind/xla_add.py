import socket
import torch
import torch_xla
import torch_xla.core.xla_model as xm

def check_metadata_dns():
    """Check if metadata.google.internal DNS record exists"""
    try:
        metadata_ip = socket.gethostbyname('metadata.google.internal')
        print(f"Found metadata.google.internal at IP: {metadata_ip}")
        return True
    except socket.gaierror:
        print("Could not resolve metadata.google.internal")
        return False

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

# Check metadata DNS first
metadata_exists = check_metadata_dns()
print(f"Metadata DNS check result: {metadata_exists}")

# Run calculation
result = simple_xla_calc()
print(f"Result shape: {result.shape}")