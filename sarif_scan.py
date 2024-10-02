import json
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Define the chat prompt template
template = """
Answer the question below.

Here is the conversation history: {context}

SARIF Report:
{sarif_report}

Question: {question}

Answer (with line number if relevant):
"""

# Define the LLM model and chat prompt
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Function to extract relevant information from SARIF JSON
def extract_info_from_sarif(sarif_json):
    results = sarif_json['runs'][0]['results']
    messages = []
    for result in results:
        rule_id = result['ruleId']
        message = result['message']['text']
        location = result['locations'][0]['physicalLocation']
        file_uri = location['artifactLocation']['uri']
        start_line = location['region']['startLine']
        end_column = location['region']['endColumn']
        messages.append(f"Rule: {rule_id}, Message: '{message}', File: {file_uri}, Line: {start_line}, End Column: {end_column}")
    return "\n".join(messages)

def handle_conversation():
    context = ""
    print("Welcome, this is an AI ChatBot for testing with SARIF output!")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        #Modify the path to the wanted sarif file.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sarif_path = os.path.join(current_dir, "test1.sarif")

        try:
            with open(sarif_path, 'r') as sarif_file:
                sarif_data = json.load(sarif_file)
                sarif_report = extract_info_from_sarif(sarif_data)
        except FileNotFoundError:
            print("SARIF file not found. Please make sure the SARIF file path is correct.")
            continue

        # Pass the context and SARIF report to the model
        res = chain.invoke({"context": context, "sarif_report": sarif_report, "question": user_input})
        print("Bot: ", res)
        context += f"\nUser: {user_input}\nAI: {res}"

if __name__ == "__main__":
    handle_conversation()
