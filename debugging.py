import requests
import logging

logger = logging.getLogger(__name__)

def check_gcp_metadata():
    """
    Checks the GCP metadata server and prints the response.
    
    Returns:
        bool: True if metadata server is accessible, False otherwise
    """
    metadata_server_url = "http://metadata.google.internal/computeMetadata/v1"
    headers = {
        "Metadata-Flavor": "Google"  # Required header for GCP metadata server
    }
    
    try:
        response = requests.get(metadata_server_url, headers=headers, timeout=5)
        print(f"Metadata server response status: {response.status_code}")
        print(f"Metadata server response:\n{response.text}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to access metadata server: {str(e)}")
        print(f"Error accessing metadata server: {str(e)}")
        return False

# Try accessing the metadata server
check_gcp_metadata()
