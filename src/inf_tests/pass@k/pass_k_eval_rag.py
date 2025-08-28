import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from datetime import datetime
import glob
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# how many gens per test
pass_k_num = 3

model_path = "../../outputs-full/checkpoint-110"

# RAG Configuration
VECTOR_DB_PATH = "../../../../dataset-QA-prep/rag/vectordb/"  # Correct path to your vector database
# COLLECTION_NAME = "java_projects"
# COLLECTION_NAME = "java_projects_extra"
# COLLECTION_NAME = "java_projects_extra_nometadata"
# COLLECTION_NAME = "java_projects_extra_nometadata_leetcode_alone"
COLLECTION_NAME = "java_langchain_splitter111"




# Initialize embedding model for RAG
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def discover_test_pairs() -> Dict[str, Dict[str, str]]:
    """Discover all test pairs in the tests directory"""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        return {}
    
    test_pairs = {}
    
    # Find all .txt files (prompts)
    prompt_files = glob.glob(str(tests_dir / "*.txt"))
    
    for prompt_file in prompt_files:
        prompt_path = Path(prompt_file)
        test_name = prompt_path.stem  # e.g., "CalculatorTest" from "CalculatorTest.txt"
        
        # Look for corresponding .java file
        java_file = tests_dir / f"{test_name}.java"
        
        if java_file.exists():
            # Extract class name (remove "Test" suffix if present)
            class_name = test_name
            if class_name.endswith("Test"):
                class_name = class_name[:-4]  # Remove "Test" suffix
            
            test_pairs[test_name] = {
                "prompt_file": str(prompt_path),
                "test_file": str(java_file),
                "class_name": class_name,
                "test_class_name": test_name
            }
    
    return test_pairs

def list_available_tests() -> None:
    """List all available test pairs"""
    test_pairs = discover_test_pairs()
    
    if not test_pairs:
        print("No test pairs found in tests/ directory")
        return
    
    print("Available test pairs:")
    for i, (test_name, info) in enumerate(test_pairs.items(), 1):
        print(f"  {i}. {test_name} (class: {info['class_name']})")
        print(f"     Prompt: {info['prompt_file']}")
        print(f"     Test: {info['test_file']}")
        print()

def get_test_info(test_name: str = None) -> List[Dict[str, str]]:
    """Get test information for specific test(s) or all tests if none specified"""
    test_pairs = discover_test_pairs()
    
    if not test_pairs:
        raise ValueError("No test pairs found in tests/ directory")
    
    if test_name:
        if test_name in test_pairs:
            return [test_pairs[test_name]]
        else:
            raise ValueError(f"Test '{test_name}' not found. Available tests: {list(test_pairs.keys())}")
    
    # If no test specified, return all test pairs
    print(f"No specific test provided. Running all {len(test_pairs)} test pairs:")
    list_available_tests()
    return list(test_pairs.values())

def get_rag_context(query: str, top_k: int = 4, enable_rag: bool = True):
    """Get relevant context from RAG database"""
    if not enable_rag:
        print("RAG is disabled")
        return ""
        
    try:
        # Check if vector database exists
        if not os.path.exists(VECTOR_DB_PATH):
            print(f"Vector database not found at: {VECTOR_DB_PATH}")
            return ""
        
        print(f"Loading vector database from: {VECTOR_DB_PATH}")
        
        # Load vector database
        db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = db.get_or_create_collection(name=COLLECTION_NAME)
        
        # Check collection size
        collection_count = collection.count()
        print(f"Collection '{COLLECTION_NAME}' has {collection_count} documents")
        
        if collection_count == 0:
            print("Collection is empty - no RAG context available")
            return ""
        
        print(f"Querying for: {query[:100]}...")
        query_embedding = embed_model.encode([query])
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        if results['documents'] and results['documents'][0]:
            context = "\n".join(results['documents'][0])
            print(f"\n--- RAG Context Retrieved (top {len(results['documents'][0])} results) ---")
            print(context[:500] + "..." if len(context) > 500 else context)
            print("--- End RAG Context ---\n")
            return context
        else:
            print("Query returned no results")
            return ""
    except Exception as e:
        print(f"Error retrieving RAG context: {e}")
        import traceback
        traceback.print_exc()
        return ""

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    
    max_seq_length = 512
    dtype = None
    load_in_4bit = True
    
    # model_path = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama",
    )
    
    return model, tokenizer, model_path

def read_prompt(prompt_file: str) -> str:
    """Read the prompt from specified file"""
    with open(prompt_file, "r") as f:
        return f.read().strip()

def generate_solution(model, tokenizer, prompt: str, temperature: float = 0.8, enable_rag: bool = True) -> str:
    """Generate a single Java solution with RAG context"""
    # Get RAG context
    context = get_rag_context(prompt, enable_rag=enable_rag)
    
    # Create full prompt with context if available
    if context:
        full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
    else:
        full_prompt = prompt
    
    messages = [{
        "role": "user",
        "content": full_prompt
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

def extract_java_code(generated_text: str, class_name: str) -> str:
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
        if f'public class {class_name}' in line or f'class {class_name}' in line:
            class_start = i
            break
    
    if class_start >= 0:
        return '\n'.join(lines[class_start:])
    
    # Return as-is if no patterns found
    return generated_text

def test_java_solution(java_code: str, test_info: Dict[str, str], solution_id: int) -> Tuple[bool, str]:
    """Test a Java solution against JUnit tests"""
    try:
        # Create temporary directory for this solution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write the Java code
            class_file = temp_path / f"{test_info['class_name']}.java"
            with open(class_file, 'w') as f:
                f.write(java_code)
            
            # Copy the test file
            test_source = Path(test_info['test_file'])
            test_dest = temp_path / f"{test_info['test_class_name']}.java"
            if test_source.exists():
                with open(test_source, 'r') as f:
                    test_content = f.read()
                with open(test_dest, 'w') as f:
                    f.write(test_content)
            else:
                return False, f"Test file not found: {test_source}"
            
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
                str(class_file),
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

def load_results_file(filepath: str) -> dict:
    """Load existing results from JSON file, return empty structure if file doesn't exist"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load existing results from {filepath}, starting fresh")
    
    return {
        "runs": [],
        "summary": {
            "total_runs": 0,
            "models_tested": [],
            "tests_run": []
        }
    }

def save_results_to_file(filepath: str, new_results: dict) -> None:
    """Append new results to existing JSON file"""
    # Load existing results
    all_results = load_results_file(filepath)
    
    # Add new run
    all_results["runs"].append(new_results)
    all_results["summary"]["total_runs"] += 1
    
    # Update summary
    if new_results["model_path"] not in all_results["summary"]["models_tested"]:
        all_results["summary"]["models_tested"].append(new_results["model_path"])
    
    for test_result in new_results["test_results"]:
        test_name = test_result["test_name"]
        if test_name not in all_results["summary"]["tests_run"]:
            all_results["summary"]["tests_run"].append(test_name)
    
    # Save updated results
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results appended to: {filepath}")
    print(f"Total runs in file: {all_results['summary']['total_runs']}")

def run_pass_k_evaluation(test_name: str = None, k: int = pass_k_num, temperature: float = 0.8, enable_rag: bool = True) -> Tuple[dict, List[str]]:
    """Run complete pass@k evaluation for one or all tests"""
    print(f"Running pass@{k} evaluation...")
    print(f"Temperature: {temperature}")
    
    # Print RAG configuration
    if enable_rag:
        print(f"RAG: Enabled")
        print(f"Vector Database: {VECTOR_DB_PATH}")
        print(f"Collection: {COLLECTION_NAME}")
        if not os.path.exists(VECTOR_DB_PATH):
            print(f"WARNING: Vector database not found at {VECTOR_DB_PATH}")
    else:
        print(f"RAG: Disabled")
    
    # Get test information (now returns a list)
    test_infos = get_test_info(test_name)
    
    # Load model once
    print("Loading model...")
    model, tokenizer, model_path = load_model_and_tokenizer()
    
    all_results = []
    
    for test_info in test_infos:
        print(f"\n{'='*60}")
        print(f"Running test: {test_info['test_class_name']} (class: {test_info['class_name']})")
        print(f"{'='*60}")
        
        # Read prompt
        prompt = read_prompt(test_info['prompt_file'])
        print(f"Prompt: {prompt[:100]}...")
        
        results = []
        solutions = []
        
        print(f"\nGenerating and testing {k} solutions...")
        
        for i in range(k):
            print(f"Solution {i+1}/{k}:", end=" ")
            
            # Generate solution
            try:
                generated = generate_solution(model, tokenizer, prompt, temperature, enable_rag)
                java_code = extract_java_code(generated, test_info['class_name'])
                solutions.append(java_code)
                
                # Test solution
                passed, output = test_java_solution(java_code, test_info, i)
                results.append(passed)
                
                status = "PASS" if passed else "FAIL"
                print(f"{status}")
                
                if not passed:
                    print(f"  Error: {output[:200]}...")
                    
            except Exception as e:
                print(f"ERROR: {str(e)}")
                results.append(False)
                solutions.append("")
        
        # Calculate metrics for this test
        pass_counts = sum(results)
        pass_at_1 = calculate_pass_at_k(results, 1)
        pass_at_k_val = calculate_pass_at_k(results, k)
        
        test_metrics = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_info['test_class_name'],
            "class_name": test_info['class_name'],
            "model_path": model_path,
            "k": k,
            "temperature": temperature,
            "rag_enabled": enable_rag,
            "vector_db_path": VECTOR_DB_PATH if enable_rag else None,
            "total_solutions": len(results),
            "passed_solutions": pass_counts,
            "pass_at_1": pass_at_1,
            f"pass_at_{k}": pass_at_k_val,
            "pass_rate": pass_counts / len(results) if results else 0,
            "individual_results": results,
            "solutions": solutions
        }
        
        all_results.append(test_metrics)
        
        # Print results for this test
        print(f"\nResults for {test_info['test_class_name']}:")
        print(f"  Solutions that passed: {pass_counts}/{len(results)}")
        print(f"  Pass@1: {pass_at_1:.2%}")
        print(f"  Pass@{k}: {pass_at_k_val:.2%}")
        print(f"  Overall pass rate: {test_metrics['pass_rate']:.2%}")
    
    # Calculate overall metrics across all tests
    total_solutions = sum(r['total_solutions'] for r in all_results)
    total_passed = sum(r['passed_solutions'] for r in all_results)
    overall_pass_rate = total_passed / total_solutions if total_solutions > 0 else 0
    
    overall_metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "k": k,
        "temperature": temperature,
        "rag_enabled": enable_rag,
        "vector_db_path": VECTOR_DB_PATH if enable_rag else None,
        "total_tests": len(all_results),
        "total_solutions": total_solutions,
        "total_passed": total_passed,
        "overall_pass_rate": overall_pass_rate,
        "test_results": all_results
    }
    
    return overall_metrics, []

def main():
    parser = argparse.ArgumentParser(description="Run pass@k evaluation for Java programs with RAG. If no specific test is provided, runs all available tests.")
    parser.add_argument("-t", "--test", type=str, help="Specific test to run (e.g., 'CalculatorTest'). If not provided, runs all tests.")
    parser.add_argument("-k", type=int, default=pass_k_num, help="Number of solutions to generate (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature (default: 0.8)")
    parser.add_argument("-o", "--output", type=str, help="Output file for results (default: 'pass_k_results_all.json')")
    parser.add_argument("--list", action="store_true", help="List available tests and exit")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG context retrieval")
    
    args = parser.parse_args()
    
    # List tests if requested
    if args.list:
        list_available_tests()
        return 0
    
    # Check if tests directory exists
    if not os.path.exists("tests"):
        print("ERROR: tests/ directory not found")
        return 1
    
    # Check for vector database if RAG is enabled
    if not args.no_rag:
        if not os.path.exists(VECTOR_DB_PATH):
            print(f"WARNING: Vector database not found at {VECTOR_DB_PATH}")
            print("RAG context will be disabled for this run.")
            print("Use --no-rag to suppress this warning.")
        else:
            print(f"Using RAG with vector database at: {VECTOR_DB_PATH}")
    else:
        print("RAG context retrieval disabled.")
    
    # Get test info (will prompt if multiple tests and none specified)
    try:
        test_infos = get_test_info(args.test)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Check if required files exist for all tests
    for test_info in test_infos:
        if not os.path.exists(test_info['prompt_file']):
            print(f"ERROR: Prompt file not found: {test_info['prompt_file']}")
            return 1
        
        if not os.path.exists(test_info['test_file']):
            print(f"ERROR: Test file not found: {test_info['test_file']}")
            return 1
    
    # Generate output filename if not specified (use a persistent filename)
    if not args.output:
        args.output = "pass_k_results_all.json"
    
    # Run evaluation
    try:
        metrics, _ = run_pass_k_evaluation(args.test, args.k, args.temperature, not args.no_rag)
        
        # Print overall results
        print(f"\n{'='*60}")
        print(f"OVERALL PASS@K EVALUATION RESULTS")
        print(f"Model: {metrics['model_path']}")
        print(f"RAG: {'Enabled' if metrics['rag_enabled'] else 'Disabled'}")
        if metrics['rag_enabled'] and metrics['vector_db_path']:
            print(f"Vector DB: {metrics['vector_db_path']}")
        print(f"{'='*60}")
        print(f"Total tests run: {metrics['total_tests']}")
        print(f"Total solutions generated: {metrics['total_solutions']}")
        print(f"Total solutions that passed: {metrics['total_passed']}")
        print(f"Overall pass rate: {metrics['overall_pass_rate']:.2%}")
        
        # Print individual test results
        print(f"\nIndividual test results:")
        for test_result in metrics['test_results']:
            print(f"  {test_result['test_name']}:")
            print(f"    Pass@1: {test_result['pass_at_1']:.2%}")
            print(f"    Pass@{args.k}: {test_result[f'pass_at_{args.k}']:.2%}")
            print(f"    Pass rate: {test_result['pass_rate']:.2%}")
        
        # Save results to persistent file
        save_results_to_file(args.output, metrics)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
