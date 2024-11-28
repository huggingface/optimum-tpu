import socket
import requests

def check_metadata_dns():
    """Check if metadata.google.internal DNS record exists"""
    try:
        metadata_ip = socket.gethostbyname('metadata.google.internal')
        print(f"Found metadata.google.internal at IP: {metadata_ip}")
        return True
    except socket.gaierror:
        print("Could not resolve metadata.google.internal")
        return False

# Check metadata DNS first
metadata_exists = check_metadata_dns()
print(f"Metadata DNS check result: {metadata_exists}")

# Check metadata server with requests
try:
    headers = {'Metadata-Flavor': 'Google'}
    response = requests.get('http://metadata.google.internal/computeMetadata/v1/instance/image', headers=headers)

    print(f"Trying to manually access metadata server: Metadata server status code: {response.status_code}")
    print(f"Trying to manually access metadata server: Metadata server response: {response.text}")
    
    if response.status_code != 200:
        raise Exception(f"Trying to manually access metadata server: Metadata server returned status code {response.status_code}")

except Exception as e:
    print(f"Trying to manually access metadata server: Error accessing metadata server: {e}")

import torch
# import torch_xla
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