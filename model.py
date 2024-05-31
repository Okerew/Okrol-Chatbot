import os
import torch
import spacy
import json
import re
from train_transformer import TransformerChatBot

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the vocabulary
with open('model/vocab.json', 'r') as f:
    vocab = json.load(f)

# Load the maximum length of the input sequences
with open('model/max_length.txt', 'r') as f:
    max_length = int(f.read())

# Define SOS and EOS tokens
SOS_token = vocab['<SOS>']
EOS_token = vocab['<EOS>']

chatbot = TransformerChatBot(
    vocab_size=len(vocab),
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    max_seq_length=max_length
)
try:
    chatbot.load_state_dict(torch.load('model/transformer_chatbot.pt', map_location=torch.device('cpu')))
except RuntimeError as e:
    print(f"Error loading model state dict: {e}")
chatbot = chatbot.to('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_expression(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return None


def generate_response(chatbot, input_sequence, vocab, max_length):
    chatbot.eval()
    input_tensor = torch.tensor([input_sequence], dtype=torch.long)
    target_sequence = [SOS_token]

    with torch.no_grad():
        for _ in range(max_length):
            target_tensor = torch.tensor([target_sequence], dtype=torch.long)
            output = chatbot(input_tensor, target_tensor)
            next_token = output.argmax(2)[:, -1].item()
            target_sequence.append(next_token)
            if next_token == EOS_token:
                break

    response_tokens = [list(vocab.keys())[list(vocab.values()).index(token)] for token in target_sequence[1:] if token in vocab.values()]
    return response_tokens


def interact_with_chatbot(chatbot, vocab, max_length):
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

        # Searches the user input for mathematical operations
        operation = re.search(r'\d+\s*[\+\*\-\/]\s*\d+', user_input)
        if operation:
            result = evaluate_expression(operation.group())
        else:
            result = ''

        # Lowercase and trim user input
        user_input = user_input.lower().strip()

        # Tokenize user input
        doc = nlp(user_input)
        tokens = [token.text for token in doc]

        # Convert tokens to sequence of integers
        sequence = [vocab.get(word, vocab['<UNK>']) for word in tokens]

        # Add SOS token and pad sequence
        sequence = [SOS_token] + sequence
        padded_sequence = sequence + [0] * (max_length - len(sequence))

        # Generate response
        response_tokens = generate_response(chatbot, padded_sequence, vocab, max_length)

        # Record interaction
        interaction = {
            "text": user_input,
            "response": ' '.join(response_tokens),
        }
        interactions.append(interaction)

        print("Chatbot's response:", ' '.join(response_tokens), result)

    # Save interactions to a JSON file
    interaction_file = 'user_interaction.json'
    with open(f'training_data/{interaction_file}', 'w') as f:
        json.dump(interactions, f)

    print("Interactions saved to:", interaction_file)


# Interact with the chatbot and save interactions to a JSON file
interact_with_chatbot(chatbot, vocab, max_length)
