import json
import sys

def prepare_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for question, responses in data.items():
            for model, response in responses.items():
                samples.append({
                    "prompt": question,
                    "response": response,
                    "model": model
                })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Successfully processed {len(samples)} samples.")
        print(f"Output written to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        print("Please make sure the file exists in the correct location.")
    except json.JSONDecodeError as e:
        print(f"Error: The file '{input_file}' is not a valid JSON file.")
        print(f"JSON error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_data.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    prepare_data(input_file, output_file)

