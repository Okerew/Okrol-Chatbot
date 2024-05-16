import os
import torch
import spacy
import json
import re
from train import padded_input_sequences

# Load the chatbot and vocabulary
chatbot = torch.load('model/chatbot.pt')
# Sets the chatbot to use cpu as I developed it on macOS
chatbot = chatbot.to('cpu')
with open('model/vocab.json', 'r') as f:
    vocab = json.load(f)

# Load the maximum length of the input sequences
with open('model/max_length.txt', 'r') as f:
    max_length = int(f.read())

# Define SOS and EOS tokens
SOS_token = vocab['<SOS>']
EOS_token = vocab['<EOS>']

def evaluate_expression(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return None

# Function to interact with the chatbot
"""
    Interacts with the chatbot by processing user input, generating responses, and saving interactions to a JSON file.

    Parameters:
    - chatbot: The chatbot model to interact with.
    - vocab: The vocabulary mapping words to integers for tokenization.
    
    Returns:
    - Model response
    - A generated file from the users interactions

"""
def interact_with_chatbot(chatbot, vocab):
    interactions = []
    while True:
        print("Type ?help for a list of available commands.")
        user_input = input("Input your text: ")
        if user_input.lower() == "?quit":
            break
        elif user_input.lower() == "?help":
            print("Available commands:")
            print("?help - Show this help message")
            print("?quit - Quit the program")
            print("?clear - Clear the console")
            continue
        elif user_input.lower() == "?clear":
            os.system('clear')
            continue
            
        # Searches the user input for mathematical operations as they always follow the same rules 
        # So it is not needed to train the model on them
        operation = re.search(r'\d+\s*[\+\*\-\/]\s*\d+', user_input)
        if operation:
            result = evaluate_expression(operation.group())
        else:
            result = ''

        # Lowercase and trim user input
        user_input = user_input.lower().strip()

        # Tokenize user input
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(user_input)
        tokens = [token.text for token in doc]

        # Convert tokens to sequence of integers
        sequence = []
        for word in tokens:
            if word in vocab:
                sequence.append(vocab[word])
            else:
                sequence.append(vocab['<UNK>'])

        # Pad sequence
        max_length = max([len(sequence) for sequence in padded_input_sequences])
        padded_sequence = sequence + [0] * (max_length - len(sequence))

        # Convert to tensor and change data type to long
        input_tensor = torch.tensor([padded_sequence], dtype=torch.long)

        # Reshape input tensor to max_length x 1
        input_tensor = input_tensor.squeeze(0).unsqueeze(1)

        # Feed forward through the chatbot
        with torch.no_grad():
            output = chatbot(input_tensor)

        # Get the predicted sequence of words
        predicted_sequence = []
        top_i = output.topk(1)[1]
        predicted_sequence.append(top_i[0].item())
        for i in range(1, max_length):
            predicted_sequence.append(top_i[i].item())

        # Convert the predicted sequence of integers to words
        predicted_response = []
        for i in predicted_sequence:
            if i == 0:
                break
            predicted_response.append(list(vocab.keys())[list(vocab.values()).index(i)])

        # Record interaction
        interaction = {
            "text": user_input,
            "response": ' '.join(predicted_response),
        }
        interactions.append(interaction)

        print("Chatbot's response:", ' '.join(predicted_response), result)

    # Save interactions to a JSON file
    interaction_file = 'user_interaction.json'
    with open(f'training_data/{interaction_file}', 'w') as f:
        json.dump(interactions, f)

    print("Interactions saved to:", interaction_file)

# Interact with the chatbot and save interactions to a JSON file
interact_with_chatbot(chatbot, vocab)
