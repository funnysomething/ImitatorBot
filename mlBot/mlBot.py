import torch
import torch.nn as nn
import pickle

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

def generate_text(model, word_to_index, seed_text, max_length):
    model.eval()  # Set the model to evaluation mode
    index_to_word = {index: word for word, index in word_to_index.items()}
    with torch.no_grad():
        seed_indices = [word_to_index[word] for word in seed_text.split()]

        for _ in range(max_length):
            input_tensor = torch.tensor(seed_indices, dtype=torch.long).view(1, -1)

            # Get the model prediction for the next word
            output = model(input_tensor)

            # Sample the next word index based on the model output probabilities
            next_word_index = torch.multinomial(torch.softmax(output, dim=-1), num_samples=1).item()

            # Add the predicted word index to the seed indices
            seed_indices.append(next_word_index)

            # Break if the generated sequence reaches the specified maximum length
            if next_word_index == word_to_index['<EOS>']:
                break

    generated_text = [index_to_word[index] for index in seed_indices]

    return ' '.join(generated_text)

def load_model():
    global TextGenerationModel
    with open("word_to_index.pkl", 'rb') as f:
        loaded_word_to_index = pickle.load(f)

    vocab = set(loaded_word_to_index.keys())
    vocab.add('<EOS>')

    loaded_word_to_index['<EOS>'] = len(vocab)-1

    loaded_model = torch.load("trained_model.pth")

    return loaded_model, loaded_word_to_index
