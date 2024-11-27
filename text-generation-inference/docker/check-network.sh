#!/bin/bash

echo "=== Starting Network Connectivity Check ==="

# Function to print section headers
print_header() {
    echo -e "\n=== $1 ==="
}

# Function to check command availability
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# Check DNS resolution
check_dns() {
    print_header "DNS Resolution Check"
    
    local domains=("github.com" "google.com" "huggingface.co")
    local success=0
    
    for domain in "${domains[@]}"; do
        echo -n "Resolving $domain... "
        if host "$domain" &> /dev/null; then
            echo "SUCCESS"
        else
            echo "FAILED"
            success=1
        fi
    done
    
    return $success
}

# Check basic connectivity
check_ping() {
    print_header "Basic Connectivity Check"
    
    local targets=("8.8.8.8" "1.1.1.1")
    local success=0
    
    for target in "${targets[@]}"; do
        echo -n "Pinging $target... "
        if ping -c 2 -W 5 "$target" &> /dev/null; then
            echo "SUCCESS"
        else
            echo "FAILED"
            success=1
        fi
    done
    
    return $success
}

# Check HTTPS connectivity
check_https() {
    print_header "HTTPS Connectivity Check"
    
    local urls=(
        "https://github.com"
        "https://huggingface.co"
        "https://api.github.com"
        "https://github.com/protocolbuffers/protobuf/releases"
    )
    local success=0
    
    for url in "${urls[@]}"; do
        echo -n "Connecting to $url... "
        if curl --max-time 10 -sI "$url" &> /dev/null; then
            echo "SUCCESS"
        else
            echo "FAILED"
            success=1
        fi
    done
    
    return $success
}

# Check required tools
print_header "Checking Required Tools"
required_tools=("curl" "ping" "host")
tools_missing=false

for tool in "${required_tools[@]}"; do
    if ! check_command "$tool"; then
        tools_missing=true
    fi
done

if [ "$tools_missing" = true ]; then
    echo "Required tools are missing. Please install them first."
    exit 1
fi

# Run all checks
failures=0

check_dns || ((failures++))
check_ping || ((failures++))
check_https || ((failures++))

print_header "Summary"
if [ $failures -eq 0 ]; then
    echo "All network checks passed successfully!"
    exit 0
else
    echo "Some network checks failed. Please review the output above."
    exit 1
fi