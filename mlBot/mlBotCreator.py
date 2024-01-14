import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

def load_data_from_sqlite(database_path):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute(f'SELECT * FROM messages')
    rawMessages = cursor.fetchall()
    connection.close()
    messages = [message[2] for message in rawMessages]
    return messages

def preprocess_text_data(text_data):
    textToString = ' '.join(text_data)
    splitWords = list(textToString)
    vocab = set(splitWords)

    word_to_index = {word: index for index, word in enumerate(vocab)}

    sequences = []
    sequence_length = 30

    for i in range(sequence_length, len(splitWords)):
        sequence = splitWords[i-sequence_length:i]
        target_word = splitWords[i]
        sequences.append((sequence, target_word))

    return word_to_index, sequences, sequence_length, vocab

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

messages = load_data_from_sqlite("messages.db")
word_to_index, sequences, sequence_length, vocab = preprocess_text_data(messages)

input_sequences = [torch.tensor([word_to_index[word] for word in seq]) for seq, _ in sequences]
target_words = [word_to_index[target] for _, target in sequences]
input_data = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True)
target_data = torch.tensor(target_words)
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Creating a new model
vocab_size = len(vocab)
embedding_dim = 50  # You can adjust this value
hidden_dim = 100  # You can adjust this value
new_model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim)

# Training the new model
optimizer = optim.Adam(new_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 20  # You can adjust this value
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = new_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

model_path = 'trained_model.pth'
word_to_index_path = 'word_to_index.pkl'

torch.save(new_model, model_path)

with open(word_to_index_path, 'wb') as f:
    pickle.dump(word_to_index, f)