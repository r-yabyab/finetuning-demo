import json

def convert_qa_to_roles(input_file, output_file):
    """
    Convert question-answer pairs to role-based format for chat training.
    
    Args:
        input_file (str): Path to input JSON file with question-answer pairs
        output_file (str): Path to output JSON file with role-based format
    """
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    # Convert each question-answer pair
    for item in data:
        if 'question' in item and 'answer' in item:
            # Create conversation pair
            conversation = [
                {
                    "role": "user",
                    "content": item['question']
                },
                {
                    "role": "assistant", 
                    "content": item['answer']
                }
            ]
            converted_data.append(conversation)
    
    # Create the final structure with conversations key
    final_data = {
        "conversations": converted_data
    }
    
    # Write the converted data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} question-answer pairs")
    print(f"Output saved to: {output_file}")

def convert_qa_to_roles_flat(input_file, output_file):
    """
    Alternative version that creates a flat list of role messages.
    
    Args:
        input_file (str): Path to input JSON file with question-answer pairs
        output_file (str): Path to output JSON file with flat role-based format
    """
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    # Convert each question-answer pair to separate role messages
    for item in data:
        if 'question' in item and 'answer' in item:
            # Add user message
            converted_data.append({
                "role": "user",
                "content": item['question']
            })
            
            # Add assistant message
            converted_data.append({
                "role": "assistant", 
                "content": item['answer']
            })
    
    # Create the final structure with conversations key
    final_data = {
        "conversations": converted_data
    }
    
    # Write the converted data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} question-answer pairs to {len(converted_data)} role messages")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Example usage - updated paths
    input_file = "../data/instruction-copy.json"
    
    # Option 1: JSONL format for training
    output_file_jsonl = "../data/instruction-training.jsonl"
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Write each conversation as a separate JSON line for training
    with open(output_file_jsonl, 'w', encoding='utf-8') as f:
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
    
    print(f"Converted {len(data)} question-answer pairs to JSONL training format")
    print(f"Training file saved to: {output_file_jsonl}")
    
    # Also create the grouped format for reference
    output_file_grouped = "../data/instruction-roles-grouped.json"
    convert_qa_to_roles(input_file, output_file_grouped)
    
    # And flat format
    output_file_flat = "../data/instruction-roles-flat.json"
    convert_qa_to_roles_flat(input_file, output_file_flat)
    
    print("\nConversion complete!")
    print("Two formats created:")
    print(f"1. Grouped conversations: {output_file_grouped}")
    print(f"2. Flat role messages: {output_file_flat}")
