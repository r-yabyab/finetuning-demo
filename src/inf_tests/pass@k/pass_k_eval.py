import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Tuple
import argparse
from datetime import datetime

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    
    max_seq_length = 512
    dtype = None
    load_in_4bit = True
    
    MODEL_PATH = "../outputs-full/checkpoint-110"
    # MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama",
    )
    
    return model, tokenizer

def read_prompt() -> str:
    """Read the prompt from gen_prompt.txt"""
    with open("gen_prompt.txt", "r") as f:
        return f.read().strip()

def generate_solution(model, tokenizer, prompt: str, temperature: float = 0.8) -> str:
    """Generate a single Java solution"""
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        temperature=temperature,
        top_p=0.95,
        top_k=64,
        max_new_tokens=512,
        use_cache=True,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
    assistant_response = generated_text[len(prompt_text):].strip()
    
    return assistant_response

def extract_java_code(generated_text: str) -> str:
    """Extract Java code from generated text, handling markdown code blocks"""
    # Try to find code within ```java or ``` blocks
    lines = generated_text.split('\n')
    in_code_block = False
    code_lines = []
    
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # If no code blocks found, try to extract based on class definition
    class_start = -1
    for i, line in enumerate(lines):
        if 'public class Calculator' in line or 'class Calculator' in line:
            class_start = i
            break
    
    if class_start >= 0:
        return '\n'.join(lines[class_start:])
    
    # Return as-is if no patterns found
    return generated_text

def test_java_solution(java_code: str, solution_id: int) -> Tuple[bool, str]:
    """Test a Java solution against JUnit tests"""
    try:
        # Create temporary directory for this solution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write the Java code
            calculator_file = temp_path / "Calculator.java"
            with open(calculator_file, 'w') as f:
                f.write(java_code)
            
            # Copy the test file
            test_source = Path("CalculatorTest.java")
            test_dest = temp_path / "CalculatorTest.java"
            if test_source.exists():
                with open(test_source, 'r') as f:
                    test_content = f.read()
                with open(test_dest, 'w') as f:
                    f.write(test_content)
            else:
                return False, "CalculatorTest.java not found"
            
            # Check if JUnit jar exists
            junit_jar = "junit-platform-console-standalone-1.10.0.jar"
            if not os.path.exists(junit_jar):
                return False, f"JUnit jar not found: {junit_jar}"
            
            # Get absolute path to JUnit jar
            junit_jar_abs = os.path.abspath(junit_jar)
            
            # Compile the Java files with proper classpath
            compile_cmd = [
                "javac", 
                "-cp", f".:{junit_jar_abs}",
                str(calculator_file),
                str(test_dest)
            ]
            
            compile_result = subprocess.run(
                compile_cmd, 
                cwd=temp_dir,
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                return False, f"Compilation failed: {compile_result.stderr}"
            
            # Run the tests with proper classpath
            test_cmd = [
                "java", 
                "-cp", f".:{junit_jar_abs}",
                "org.junit.platform.console.ConsoleLauncher",
                "--scan-classpath"
            ]
            
            test_result = subprocess.run(
                test_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if tests passed (JUnit returns 0 for success)
            success = test_result.returncode == 0
            output = test_result.stdout + test_result.stderr
            
            return success, output
            
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"
    except Exception as e:
        return False, f"Error running test: {str(e)}"

def calculate_pass_at_k(results: List[bool], k: int) -> float:
    """Calculate pass@k metric"""
    if k > len(results):
        k = len(results)
    
    # Count how many of the first k results passed
    passed = sum(results[:k])
    return passed / k if k > 0 else 0.0

def run_pass_k_evaluation(k: int = 10, temperature: float = 0.8) -> dict:
    """Run complete pass@k evaluation"""
    print(f"Running pass@{k} evaluation...")
    print(f"Temperature: {temperature}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Read prompt
    prompt = read_prompt()
    print(f"Prompt: {prompt[:100]}...")
    
    results = []
    solutions = []
    
    print(f"\nGenerating and testing {k} solutions...")
    
    for i in range(k):
        print(f"Solution {i+1}/{k}:", end=" ")
        
        # Generate solution
        try:
            generated = generate_solution(model, tokenizer, prompt, temperature)
            java_code = extract_java_code(generated)
            solutions.append(java_code)
            
            # Test solution
            passed, output = test_java_solution(java_code, i)
            results.append(passed)
            
            status = "PASS" if passed else "FAIL"
            print(f"{status}")
            
            if not passed:
                print(f"  Error: {output[:200]}...")
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append(False)
            solutions.append("")
    
    # Calculate metrics
    pass_counts = sum(results)
    pass_at_1 = calculate_pass_at_k(results, 1)
    pass_at_k_val = calculate_pass_at_k(results, k)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "k": k,
        "temperature": temperature,
        "total_solutions": len(results),
        "passed_solutions": pass_counts,
        "pass_at_1": pass_at_1,
        f"pass_at_{k}": pass_at_k_val,
        "pass_rate": pass_counts / len(results) if results else 0,
        "individual_results": results
    }
    
    return metrics, solutions

def main():
    parser = argparse.ArgumentParser(description="Run pass@k evaluation for Java Calculator")
    parser.add_argument("-k", type=int, default=10, help="Number of solutions to generate (default: 10)")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Generation temperature (default: 0.8)")
    parser.add_argument("-o", "--output", type=str, default="pass_k_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Check if required files exist
    if not os.path.exists("gen_prompt.txt"):
        print("ERROR: gen_prompt.txt not found")
        return
    
    if not os.path.exists("CalculatorTest.java"):
        print("ERROR: CalculatorTest.java not found")
        return
    
    # Run evaluation
    try:
        metrics, solutions = run_pass_k_evaluation(args.k, args.temperature)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"PASS@K EVALUATION RESULTS ({MODEL_PATH})")
        print(f"{'='*50}")
        print(f"Total solutions generated: {metrics['total_solutions']}")
        print(f"Solutions that passed: {metrics['passed_solutions']}")
        print(f"Pass@1: {metrics['pass_at_1']:.2%}")
        print(f"Pass@{args.k}: {metrics[f'pass_at_{args.k}']:.2%}")
        print(f"Overall pass rate: {metrics['pass_rate']:.2%}")
        
        # Save detailed results
        detailed_results = {
            "metrics": metrics,
            "solutions": solutions
        }
        
        with open(args.output, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
