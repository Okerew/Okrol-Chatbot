import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import json
import os
import random

nlp = spacy.load('en_core_web_sm')
# Function to load data from JSON files
"""
Loads data from JSON files in the specified folder_path, extracts text and response labels,
builds a vocabulary with unique words from text and response fields, and returns the processed data, labels, and vocabulary.
"""
def load_data(folder_path):
    data = []
    labels = []
    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<SOS>'] = 2
    vocab['<EOS>'] = 3
    next_index = 4

    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name), 'r') as f:
            file_contents = json.load(f)
            if isinstance(file_contents, list):
                for entry in file_contents:
                    data.append(entry['text'])
                    labels.append(entry['response'])
                    for word in nlp(entry['text'].lower()):
                        if word.text not in vocab:
                            vocab[word.text] = next_index
                            next_index += 1
                    for word in nlp(entry['response'].lower()):
                        if word.text not in vocab:
                            vocab[word.text] = next_index
                            next_index += 1
            else:
                data.append(file_contents['text'])
                labels.append(file_contents['response'])
                for word in nlp(file_contents['text'].lower()):
                    if word.text not in vocab:
                        vocab[word.text] = next_index
                        next_index += 1
                for word in nlp(file_contents['response'].lower()):
                    if word.text not in vocab:
                        vocab[word.text] = next_index
                        next_index += 1

    return data, labels, vocab


# Function to tokenize text data
def tokenize_data(data):
    tokenized_data = []
    for text in data:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        tokenized_data.append(tokens)
    return tokenized_data

"""
    Convert tokenized data to sequences of integers based on a given vocabulary.

    Parameters:
    - tokenized_data (list): List of tokenized texts to convert.
    - vocab (dict): Vocabulary dictionary mapping words to integers.

    Returns:
    - sequences (list): List of sequences of integers corresponding to the tokenized data.
"""
def convert_to_sequences(tokenized_data, vocab):
    sequences = []
    for text in tokenized_data:
        sequence = []
        for word in text:
            if word in vocab:
                sequence.append(vocab[word])
            elif word not in vocab:
                vocab[word] = len(vocab)
                sequence.append(vocab[word])
        sequences.append(sequence)
    return sequences


# Function to pad sequences
def pad_sequences(sequences, max_length):
    padded = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padded.append(sequence + [0] * (max_length - len(sequence)))
        else:
            padded.append(sequence[:max_length])
    return padded


# Define the encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


"""
        Initializes the DecoderRNN module.

        Args:
            hidden_size (int): The size of the hidden state.
            output_size (int): The size of the output.
            dropout_p (float, optional): The dropout probability. Defaults to 0.1.
"""
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input).view(1, 1, -1))
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden


# Define the chatbot
"""
        Forward pass of the ChatBot model.

        Parameters:
            input_sequence (torch.Tensor): The input sequence to the model.
            target_sequence (torch.Tensor, optional): The target sequence for training. Defaults to None.
            teacher_forcing_ratio (float, optional): The probability of using teacher forcing. Defaults to 0.5.

        Returns:
            torch.Tensor: The outputs of the model.
"""


class ChatBot(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(ChatBot, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, vocab_size)

    def forward(self, input_sequence, target_sequence=None, teacher_forcing_ratio=0.5):
        input_length = input_sequence.size(0)
        if target_sequence is not None:
            target_length = target_sequence.size(0)
        else:
            target_length = input_length

        outputs = torch.zeros(target_length, vocab_size)

        hidden = self.encoder.initHidden()

        for ei in range(input_length):
            output, hidden = self.encoder(input_sequence[ei], hidden)

        decoder_input = torch.tensor([[SOS_token]], dtype=torch.long)

        for di in range(target_length):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[di] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            if target_sequence is not None and teacher_force:
                decoder_input = target_sequence[di]
            else:
                decoder_input = top1

        return outputs


# Load and tokenize text data from JSON files
folder_path = 'training_data'
data, labels, vocab = load_data(folder_path)
tokenized_data = tokenize_data(data)
tokenized_labels = tokenize_data(labels)

# Create a vocabulary
for text in tokenized_data:
    for word in text:
        if word not in vocab:
            vocab[word] = len(vocab)


SOS_token = vocab['<SOS>']
EOS_token = vocab['<EOS>']

# Convert tokenized data to sequences of integers
input_sequences = convert_to_sequences(tokenized_data, vocab)
target_sequences = convert_to_sequences(tokenized_labels, vocab)

# Add SOS and EOS tokens to the sequences
for i in range(len(input_sequences)):
    input_sequences[i].insert(0, SOS_token)
    target_sequences[i].append(EOS_token)

# Pad sequences
max_length = max([len(sequence) for sequence in input_sequences])
padded_input_sequences = pad_sequences(input_sequences, max_length)
padded_target_sequences = pad_sequences(target_sequences, max_length)

# Instantiate the chatbot
vocab_size = len(vocab)
hidden_size = 512 
chatbot = ChatBot(vocab_size, hidden_size)

# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss
optimizer = optim.Adam(chatbot.parameters(), lr=0.001)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
learning_rate = optimizer.param_groups[0]['lr']

# Set number of epochs
num_epochs = 10

# Add dropout
dropout_p = 0.2
chatbot.encoder = EncoderRNN(vocab_size, hidden_size, dropout_p=dropout_p)
chatbot.decoder = DecoderRNN(hidden_size, vocab_size, dropout_p=dropout_p)

# Train the chatbot
for epoch in range(num_epochs):
    for sequence_pair in zip(padded_input_sequences, padded_target_sequences):
        input_sequence = torch.tensor(sequence_pair[0], dtype=torch.long)
        target_sequence = torch.tensor(sequence_pair[1], dtype=torch.long)

        optimizer.zero_grad()

        output = chatbot(input_sequence, target_sequence, teacher_forcing_ratio=1.0)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        target = target_sequence[1:].view(-1)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

    current_lr = scheduler.get_last_lr()[0]  # Access the current learning rate
    scheduler.step(loss)

    print(f"Debug: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")

# Save the chatbot
torch.save(chatbot, 'model/chatbot.pt')

# Save the vocabulary
with open('model/vocab.json', 'w') as f:
    json.dump(vocab, f)
    
