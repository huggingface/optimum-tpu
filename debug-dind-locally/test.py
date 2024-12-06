# test_setup.py
import os
import subprocess
import time

def create_dockerfile():
    dockerfile_content = """FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm

WORKDIR /app
RUN pip install fastapi uvicorn

COPY server.py .

EXPOSE 80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

def create_server():
    server_content = """from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    inputs: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    return {
        "generated_text": "Hello World!",
        "request_received": request.dict()
    }
"""
    with open("server.py", "w") as f:
        f.write(server_content)

def run_test():
    # Create necessary files
    create_dockerfile()
    create_server()

    try:
        # Build container
        print("Building container...")
        subprocess.run(["docker", "build", "-t", "test-tgi-server", "."], check=True)

        # Run container with explicit network binding
        print("Starting container...")
        subprocess.run([
            "docker", "run", "-d",
            "-p", "0.0.0.0:80:80",  # Explicitly bind to 0.0.0.0
            "--name", "test-server",
            "--network", "host",  # Use host networking
            "test-tgi-server"
        ], check=True)

        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(5)

        # Show logs
        print("Container logs:")
        subprocess.run(["docker", "logs", "test-server"], check=True)

        # Test endpoint using 0.0.0.0
        print("\nTesting endpoint...")
        curl_command = [
            "curl", "--max-time", "30", "http://0.0.0.0:80/generate",
            "-X", "POST",
            "-d", '{"inputs":"test message"}',
            "-H", "Content-Type: application/json"
        ]
        subprocess.run(curl_command, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    finally:
        # Cleanup
        print("\nCleaning up...")
        subprocess.run(["docker", "stop", "test-server"], check=False)
        subprocess.run(["docker", "rm", "test-server"], check=False)

if __name__ == "__main__":
    run_test()