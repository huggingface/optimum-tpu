import requests
import logging
import sys

logger = logging.getLogger(__name__)

def check_gcp_metadata():
    """
    Checks the GCP metadata server image information and prints the response.
    
    Returns:
        bool: True if metadata server is accessible with 200 status code
    
    Raises:
        SystemExit: If metadata server returns non-200 status code or is inaccessible
    """
    metadata_server_url = "http://metadata.google.internal/computeMetadata/v1/instance/image"
    headers = {
        "Metadata-Flavor": "Google"  # Required header for GCP metadata server
    }
    
    try:
        response = requests.get(metadata_server_url, headers=headers, timeout=5)
        print(f"Metadata server response status: {response.status_code}")
        print(f"Instance image info:\n{response.text}")
        
        if response.status_code != 200:
            logger.error(f"Metadata server returned status code {response.status_code}")
            logger.error(f"Metadata server response:\n{response.text}")
            sys.exit(1)
            
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to access metadata server: {str(e)}")
        print(f"Error accessing metadata server: {str(e)}")
        sys.exit(1)

# Try accessing the metadata server
check_gcp_metadata()
