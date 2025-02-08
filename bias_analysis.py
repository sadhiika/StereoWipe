import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json
from datasets import load_dataset

# Load the CrowS-Pairs dataset
crows_pairs_dataset = load_dataset("crows-pairs", "all")

# Load a pre-trained model for bias detection (e.g., a model trained on similar tasks)
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize the pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to evaluate a single response
def evaluate_response(response, classifier):
    result = classifier(response, truncation=True)
    return result

# Read and process the JSONL file
results = []
with open('llm_responses.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():
            item = json.loads(line)
            if 'response' in item:
                try:
                    evaluation = evaluate_response(item['response'], classifier)
                    result = {
                        'Prompt': item['prompt'],
                        'Model': item['model'],
                        'Evaluation': evaluation
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error processing response: {e}")

# Generate a report
def generate_report(results):
    report_data = []
    for result in results:
        for eval_item in result['Evaluation']:
            report_data.append({
                'Prompt': result['Prompt'],
                'Model': result['Model'],
                'Label': eval_item['label'],
                'Score': eval_item['score']
            })
    return pd.DataFrame(report_data)

bias_report = generate_report(results)
bias_report.to_csv('bias_report.csv', index=False)

# Calculate the mean bias score for each model
def calculate_mean_bias(group):
    return group['Score'].mean()

average_bias = bias_report.groupby('Model').apply(calculate_mean_bias).reset_index(name='Bias_Score')

# Plotting the bar graph with a more detailed scale
plt.figure(figsize=(10, 6))
plt.bar(average_bias['Model'], average_bias['Bias_Score'], color='skyblue')
plt.xlabel('Chatbot Model')
plt.ylabel('Average Bias Score')
plt.title('Average Bias Score by Chatbot Model')
plt.xticks(rotation=45)
plt.ylim(min(average_bias['Bias_Score']) - 0.1, max(average_bias['Bias_Score']) + 0.1)
plt.tight_layout()

plt.savefig('average_bias_score.png')
plt.show()
