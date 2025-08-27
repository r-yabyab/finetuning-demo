# Pass@K Evaluation Script

This script evaluates fine-tuned models using the pass@k metric for Java programming tasks.

## Usage

### Basic Usage
```bash
python pass_k_eval.py
```

### Specify a particular test
```bash
python pass_k_eval.py --test CalculatorTest
```

### List available tests
```bash
python pass_k_eval.py --list
```

### Custom parameters
```bash
python pass_k_eval.py --test StringUtilsTest -k 5 --temperature 0.7 -o my_results.json
```

## Arguments

- `--test, -t`: Specific test to run (e.g., 'CalculatorTest', 'StringUtilsTest')
- `-k`: Number of solutions to generate (default: 10)
- `--temperature`: Generation temperature (default: 0.8)
- `-o, --output`: Output file for results (auto-generated if not specified)
- `--list`: List available tests and exit

## Test Structure

The script looks for test pairs in the `tests/` directory:
- `{TestName}.txt`: Contains the prompt for the model
- `{TestName}.java`: Contains the JUnit tests

For example:
- `tests/CalculatorTest.txt` - prompt for Calculator class
- `tests/CalculatorTest.java` - JUnit tests for Calculator class

The script automatically extracts the class name by removing the "Test" suffix from the test name.

## Output

The script generates a JSON file with detailed results including:
- Pass@1 and Pass@k metrics
- Individual test results
- Generated solutions
- Model and test metadata

## Requirements

- Python 3.7+
- unsloth library
- JUnit Platform Console Standalone JAR file
- Java compiler (javac)
