import random
import string
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define constants
MAX_LENGTH = 500
OOD_MAX_LENGTH = 1500
VALID_CHARACTERS = ["s", "a", "b", "c", "d", "x", "e", "p"]
MAIN_CHARACTERS = ["a", "b", "c", "d", "x"]
START_TOKEN = "s"
END_TOKEN = "e"
PADDING_TOKEN = "p"
VALID_RATIO = 0.5  # Half of the dataset should be valid


# Define constants for model
VOCAB_SIZE = len(VALID_CHARACTERS)
EMBEDDING_DIM = 6
NUM_HEADS = 2
NUM_LAYERS = 1
HIDDEN_DIM = 2
BATCH_SIZE = 625
EPOCHS = 15

# Mapping characters to indices
char_to_index = {ch: idx for idx, ch in enumerate(VALID_CHARACTERS)}

# Custom dataset class
class StringDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.labels = []
        with open(file_path, "r") as f:
            for line in tqdm(f):
                parts = line.strip().split(" ")
                self.data.append(parts[0])
                self.labels.append(int(parts[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string = self.data[idx]
        label = self.labels[idx]
        encoded = self.encode_string(string)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(
            label, dtype=torch.float32
        )

    def encode_string(self, string):
        return [char_to_index[char] for char in string]


# Transformer model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, OOD_MAX_LENGTH, embedding_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(OOD_MAX_LENGTH * embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        #print('x_emb: ', x.shape)
        x = self.transformer_encoder(x)
        #print('x_feature: ', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

# Prepare dataset and dataloader
print('########')
print('Start loading training data')
print('########')
dataset = StringDataset("more_complex_dataset/train_dataset.txt")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print('########')
print('Training data loaded.')
print('########')


# Initialize model, loss function, and optimizer
model = TransformerClassifier(
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and torch.backends.mps.is_available():
    device = torch.device("mps")
model.to(device)

# Training loop
print('########')
print(f'Start training for Phase 1: {EPOCHS} epochs')
print('########')
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs.squeeze(), labels.to(device))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            outputs = model(inputs.to(device))
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted.squeeze().to(device) == labels.to(device)).sum().item()
    print(f"Accuracy: {correct/total}")
    # Save the trained model
    param_save_path = "models/test_default.pth"
    torch.save(model.state_dict(), param_save_path)
    print('########')
    print(f'model.state_dict saved at path {param_save_path}')
    print('########')
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print('########')
print(f'Start training for Phase 2: {EPOCHS} epochs')
print('########')

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs.squeeze(), labels.to(device))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted.squeeze().to(device) == labels.to(device)).sum().item()
    print(f"Accuracy: {correct/total}")

# Save the trained model
param_save_path = "short70_200total_transformer_model.pth"
torch.save(model.state_dict(), param_save_path)
print('########')
print(f'model.state_dict saved at path {param_save_path}')
print('########')
test_dataset = StringDataset("more_complex_dataset/test_dataset.txt")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Print accuracy of the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_dataloader):
        outputs = model(inputs.to(device))
        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted.squeeze().to(device) == labels.to(device)).sum().item()
print(f"Accuracy: {correct/total}")