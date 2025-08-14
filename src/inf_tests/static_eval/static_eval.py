import os
import json
from codebleu import calc_codebleu
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Configuration
MODEL_PATH = "../outputs-full/checkpoint-110"
max_seq_length = 512
dtype = None
load_in_4bit = True
temperature = 1.0
max_new_tokens = 1000

# Load model and tokenizer
print("Loading model...")
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

def generate_prediction(prompt_text):
    """Generate prediction from the model given a prompt"""
    messages = [{
        "role": "user",
        "content": prompt_text
    }]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to("cuda")
    
    # Generate without streaming for cleaner output
    outputs = model.generate(
        input_ids=inputs,
        temperature=temperature,
        top_p=0.95,
        top_k=64,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True
    )
    
    # Decode only the generated part (skip the input prompt)
    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def load_file_content(filepath):
    """Load content from a file"""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
        return ""

def evaluate_predictions():
    """Run evaluation on all prompt-reference pairs"""
    base_dir = "/home/ubuntu/finetuning-demo/src/static_eval"
    
    results = []
    
    # Find all prompt files
    prompt_files = [f for f in os.listdir(base_dir) if f.startswith('prompt') and f.endswith('.txt')]
    prompt_files.sort()
    
    for prompt_file in prompt_files:
        # Extract number from prompt file (e.g., prompt1.txt -> 1)
        prompt_num = prompt_file.replace('prompt', '').replace('.txt', '')
        ref_file = f"ref{prompt_num}.java"
        
        prompt_path = os.path.join(base_dir, prompt_file)
        ref_path = os.path.join(base_dir, ref_file)
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {prompt_file} -> {ref_file}")
        print(f"{'='*60}")
        
        # Load prompt and reference
        prompt_text = load_file_content(prompt_path)
        reference = load_file_content(ref_path)
        
        if not prompt_text or not reference:
            print(f"Skipping {prompt_file} due to missing files")
            continue
        
        print(f"Prompt: {prompt_text[:100]}...")
        
        # Generate prediction
        print("Generating prediction...")
        prediction = generate_prediction(prompt_text)
        
        print(f"Generated {len(prediction)} characters")
        print(f"Prediction preview: {prediction[:200]}...")
        
        # Calculate CodeBLEU
        try:
            result = calc_codebleu(
                [reference], 
                [prediction], 
                lang="java",  # Changed to java since references are Java files
                weights=(0.25, 0.25, 0.25, 0.25),
                tokenizer=None
            )
            
            print(f"CodeBLEU Score: {result['codebleu']:.4f}")
            print(f"  - BLEU: {result['ngram_match_score']:.4f}")
            print(f"  - Weighted NGRAM: {result['weighted_ngram_match_score']:.4f}")
            print(f"  - Syntax: {result['syntax_match_score']:.4f}")
            print(f"  - Dataflow: {result['dataflow_match_score']:.4f}")
            
            results.append({
                'prompt_file': prompt_file,
                'ref_file': ref_file,
                'prompt': prompt_text,
                'prediction': prediction,
                'reference': reference,
                'scores': result
            })
            
        except Exception as e:
            print(f"Error calculating CodeBLEU: {e}")
            results.append({
                'prompt_file': prompt_file,
                'ref_file': ref_file,
                'prompt': prompt_text,
                'prediction': prediction,
                'reference': reference,
                'error': str(e)
            })
    
    return results

def save_results(results):
    """Save detailed results to JSON file"""
    output_file = "/home/ubuntu/finetuning-demo/src/static_eval/evaluation_results.json"
    
    # Prepare summary
    summary = {
        'model_path': MODEL_PATH,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'total_evaluations': len(results),
        'successful_evaluations': len([r for r in results if 'scores' in r]),
        'average_codebleu': 0,
        'results': results
    }
    
    # Calculate average CodeBLEU
    successful_results = [r for r in results if 'scores' in r]
    if successful_results:
        avg_codebleu = sum(r['scores']['codebleu'] for r in successful_results) / len(successful_results)
        summary['average_codebleu'] = avg_codebleu
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Successful evaluations: {summary['successful_evaluations']}")
    if summary['average_codebleu'] > 0:
        print(f"Average CodeBLEU: {summary['average_codebleu']:.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    print("Starting static evaluation...")
    results = evaluate_predictions()
    save_results(results)
    print("\nEvaluation complete!")