#!/bin/bash
# Setup script for pass@k evaluation on Ubuntu

echo "Setting up Java and JUnit for pass@k evaluation..."

# Install Java if not present
if ! command -v javac &> /dev/null; then
    echo "Installing OpenJDK 21..."
    sudo apt update
    sudo apt install -y openjdk-21-jdk
else
    echo "Java is already installed"
fi

# Download JUnit if not present
JUNIT_JAR="junit-platform-console-standalone-1.10.0.jar"
if [ ! -f "$JUNIT_JAR" ]; then
    echo "Downloading JUnit..."
    wget https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.0/junit-platform-console-standalone-1.10.0.jar
else
    echo "JUnit jar already exists"
fi

# Make the evaluation script executable
chmod +x pass_k_eval.py

echo "Setup complete!"
echo ""
echo "To run pass@k evaluation:"
echo "  python3 pass_k_eval.py -k 10"
echo ""
echo "Options:"
echo "  -k N          Generate N solutions (default: 10)"
echo "  -t TEMP       Set temperature (default: 0.8)"
echo "  -o FILE       Output file (default: pass_k_results.json)"
