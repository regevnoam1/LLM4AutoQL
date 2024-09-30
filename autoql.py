import json
import sys
import os
from typing import List, Dict

from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# Define the prompt template to filter false positives
PROMPT_TEMPLATE = """
You are an expert code analysis assistant. You will be provided with a list of warnings from CodeQL.
Your task is to identify and remove any false positive warnings.

Provide the cleaned list of warnings without the false positives.

Here are the warnings:

{warnings}

Cleaned Warnings:
"""

def load_codeql_output(file_path: str) -> List[Dict]:
    """
    Load CodeQL output from a JSON file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    warnings = data.get('results', [])
    return warnings

def save_cleaned_warnings(cleaned_warnings: List[Dict], output_path: str):
    """
    Save the cleaned warnings to a JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump({"cleaned_results": cleaned_warnings}, f, indent=4)
    print(f"Cleaned warnings saved to '{output_path}'.")

def format_warnings(warnings: List[Dict]) -> str:
    """
    Format the warnings into a readable string for the LLM.
    """
    formatted = ""
    for idx, warning in enumerate(warnings, 1):
        formatted += f"Warning {idx}:\n"
        for key, value in warning.items():
            formatted += f"  {key}: {value}\n"
        formatted += "\n"
    return formatted

def filter_false_positives(warnings: List[Dict], model: OllamaLLM, prompt_template: str) -> List[Dict]:
    """
    Use the LLM to filter out false positive warnings.
    """
    formatted_warnings = format_warnings(warnings)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model
    
    response = chain.run({"warnings": formatted_warnings})
    
    try:
        cleaned_warnings = json.loads(response)
        return cleaned_warnings
    except json.JSONDecodeError:
        print("Error: Failed to parse the model's response. Returning original warnings.")
        return warnings

def main():
    if len(sys.argv) != 3:
        print("Usage: python autoql.py <input_codeql_output.json> <output_cleaned_warnings.json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("Loading CodeQL output...")
    warnings = load_codeql_output(input_path)
    print(f"Total warnings loaded: {len(warnings)}")
    
    print("Initializing LLM...")
    model = OllamaLLM(model="llama3")
    
    print("Filtering false positives...")
    cleaned_warnings = filter_false_positives(warnings, model, PROMPT_TEMPLATE)
    print(f"Warnings after filtering: {len(cleaned_warnings)}")
    
    print("Saving cleaned warnings...")
    save_cleaned_warnings(cleaned_warnings, output_path)

if __name__ == "__main__":
    main()
