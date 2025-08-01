import json

def convert_qa_to_training_format(input_file, output_file):
    """
    Convert question-answer pairs to training format for fine-tuning.
    Each conversation becomes a separate JSON object with a "messages" field.
    
    Args:
        input_file (str): Path to input JSON file with question-answer pairs
        output_file (str): Path to output JSONL file for training
    """
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Write each conversation as a separate JSON line
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            if 'question' in item and 'answer' in item:
                # Create training format with messages
                training_item = {
                    "messages": [
                        {
                            "role": "user",
                            "content": item['question']
                        },
                        {
                            "role": "assistant", 
                            "content": item['answer']
                        }
                    ]
                }
                # Write as JSON line
                f.write(json.dumps(training_item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(data)} question-answer pairs to training format")
    print(f"Output saved to: {output_file}")

def convert_qa_to_simple_format(input_file, output_file):
    """
    Alternative format that creates a simple list of conversation objects.
    
    Args:
        input_file (str): Path to input JSON file with question-answer pairs
        output_file (str): Path to output JSON file for training
    """
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    # Convert each question-answer pair
    for item in data:
        if 'question' in item and 'answer' in item:
            # Create training format with messages
            training_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": item['question']
                    },
                    {
                        "role": "assistant", 
                        "content": item['answer']
                    }
                ]
            }
            converted_data.append(training_item)
    
    # Write as regular JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} question-answer pairs to simple format")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    input_file = "data/instruction-copy.json"
    
    # Option 1: JSONL format (recommended for most training libraries)
    output_file_jsonl = "data/instruction-training.jsonl"
    convert_qa_to_training_format(input_file, output_file_jsonl)
    
    # Option 2: Simple JSON array format
    output_file_simple = "data/instruction-training.json"
    convert_qa_to_simple_format(input_file, output_file_simple)
    
    print("\nConversion complete!")
    print("Two training formats created:")
    print(f"1. JSONL format: {output_file_jsonl}")
    print(f"2. Simple JSON format: {output_file_simple}")
    print("\nRecommendation: Use the JSONL format for most training frameworks.")
