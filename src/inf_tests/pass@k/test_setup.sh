#!/bin/bash
# Simple test script to verify the setup works

echo "Testing Java compilation and JUnit execution..."

JUNIT_JAR="junit-platform-console-standalone-1.10.0.jar"

# Check if Java is available
if ! command -v javac &> /dev/null; then
    echo "ERROR: javac not found. Please install Java."
    exit 1
fi

# Check if JUnit jar exists
if [ ! -f "$JUNIT_JAR" ]; then
    echo "ERROR: JUnit jar not found. Run setup.sh first."
    exit 1
fi

# Check if test files exist
if [ ! -f "CalculatorTest.java" ]; then
    echo "ERROR: CalculatorTest.java not found."
    exit 1
fi

# Create a simple Calculator.java for testing
cat > Calculator.java << 'EOF'
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java Calculator <num1> <num2>");
            return;
        }
        
        Calculator calc = new Calculator();
        int num1 = Integer.parseInt(args[0]);
        int num2 = Integer.parseInt(args[1]);
        int result = calc.add(num1, num2);
        System.out.println(result);
    }
}
EOF

echo "Created test Calculator.java"

# Compile
echo "Compiling..."
javac -cp .:$JUNIT_JAR Calculator.java CalculatorTest.java

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "Compilation successful!"

# Run tests
echo "Running JUnit tests..."
java -jar $JUNIT_JAR -cp . --scan-classpath

if [ $? -eq 0 ]; then
    echo "✅ Tests PASSED! Setup is working correctly."
else
    echo "❌ Tests FAILED. Check the implementation."
fi

# Clean up
rm -f *.class

echo "Test complete."
