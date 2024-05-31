import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import json
import os

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to load data from JSON files
def load_data(folder_path):
    """
    Load text data and responses from JSON files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        data (list): List of text data.
        labels (list): List of corresponding responses.
        vocab (dict): Vocabulary mapping words to unique indices.
    """
    data = []
    labels = []
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
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

# Tokenize text data
def tokenize_data(data):
    """
    Tokenize and lemmatize the text data.

    Args:
        data (list): List of text data.

    Returns:
        tokenized_data (list): List of tokenized and lemmatized text data.
    """
    tokenized_data = []
    for text in data:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        tokenized_data.append(tokens)
    return tokenized_data

# Convert tokenized data to sequences of integers
def convert_to_sequences(tokenized_data, vocab):
    """
    Convert tokenized data into sequences of integers using the provided vocabulary.

    Args:
        tokenized_data (list): List of tokenized text data.
        vocab (dict): Vocabulary mapping words to unique indices.

    Returns:
        sequences (list): List of sequences of integers.
    """
    sequences = []
    for text in tokenized_data:
        sequence = [vocab.get(word, vocab['<UNK>']) for word in text]
        sequences.append(sequence)
    return sequences

# Pad sequences
def pad_sequences(sequences, max_length):
    """
    Pad or truncate sequences to a fixed length.

    Args:
        sequences (list): List of sequences of integers.
        max_length (int): Maximum length to pad/truncate sequences.

    Returns:
        padded_sequences (torch.Tensor): Tensor of padded sequences.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [0] * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences, dtype=torch.long)

# Define the encoder model
class Encoder(nn.Module):
    """
    Transformer Encoder model.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the embeddings.
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): Dimensionality of the feedforward layers.
        max_seq_length (int): Maximum sequence length.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)

    def forward(self, src):
        """
        Forward pass of the encoder model.

        Args:
            src (torch.Tensor): Source sequences.

        Returns:
            torch.Tensor: Encoded source sequences.
        """
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1)]
        output = self.transformer_encoder(src)
        return output

# Define the decoder model
class Decoder(nn.Module):
    """
    Transformer Decoder model.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the embeddings.
        nhead (int): Number of attention heads.
        num_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimensionality of the feedforward layers.
        max_seq_length (int): Maximum sequence length.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        """
        Forward pass of the decoder model.

        Args:
            tgt (torch.Tensor): Target sequences.
            memory (torch.Tensor): Encoded source sequences from the encoder.

        Returns:
            torch.Tensor: Decoded target sequences.
        """
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1)]
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        return output

# Define the full transformer model
class TransformerChatBot(nn.Module):
    """
    Full Transformer-based chatbot model using separate encoder and decoder.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the embeddings.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimensionality of the feedforward layers.
        max_seq_length (int): Maximum sequence length.
    """

    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length):
        super(TransformerChatBot, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length)
        self.decoder = Decoder(vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length)

    def forward(self, src, tgt):
        """
        Forward pass of the full transformer model.

        Args:
            src (torch.Tensor): Source sequences.
            tgt (torch.Tensor): Target sequences.

        Returns:
            torch.Tensor: Output sequences.
        """
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output


# Load and tokenize text data from JSON files
folder_path = 'training_data'
data, labels, vocab = load_data(folder_path)
tokenized_data = tokenize_data(data)
tokenized_labels = tokenize_data(labels)

# Convert tokenized data to sequences of integers
input_sequences = convert_to_sequences(tokenized_data, vocab)
target_sequences = convert_to_sequences(tokenized_labels, vocab)

# Add SOS and EOS tokens to the sequences
SOS_token = vocab['<SOS>']
EOS_token = vocab['<EOS>']
for i in range(len(input_sequences)):
    input_sequences[i].insert(0, SOS_token)
    target_sequences[i].append(EOS_token)

# Pad sequences
max_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))
padded_input_sequences = pad_sequences(input_sequences, max_length)
padded_target_sequences = pad_sequences(target_sequences, max_length)

# Save the maximum sequence length
with open('model/max_length.txt', 'w') as f:
    f.write(str(max_length))

# Instantiate the transformer chatbot
vocab_size = len(vocab)
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
chatbot = TransformerChatBot(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                             max_length)

# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token in loss computation
optimizer = optim.Adam(chatbot.parameters(), lr=0.001)

# Train the chatbot
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for input_sequence, target_sequence in zip(padded_input_sequences, padded_target_sequences):
        input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
        target_sequence = target_sequence.unsqueeze(0)  # Add batch dimension
        target_input = target_sequence[:, :-1]
        target_output = target_sequence[:, 1:]

        optimizer.zero_grad()
        output = chatbot(input_sequence, target_input)
        output_dim = output.shape[-1]

        # Flatten the output and target_output tensors
        output = output.contiguous().view(-1, output_dim)
        target_output = target_output.contiguous().view(-1)

        loss = criterion(output, target_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(padded_input_sequences)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the chatbot
torch.save(chatbot.state_dict(), 'model/transformer_chatbot.pt')

# Save the vocabulary
with open('model/vocab.json', 'w') as f:
    json.dump(vocab, f)
