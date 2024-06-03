import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
from tqdm import tqdm
from accelerate import Accelerator
import os
from loguru import logger
import time

logger.add(os.path.join('/orion/u/yrichard/n/logs', f'{time.strftime("%Y-%m-%d-%H-%M-%S")}.log'))


os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'

# Define constants
MAX_LENGTH = 200
OOD_MAX_LENGTH = 400

logger.info(f'MAX_LENGTH: {MAX_LENGTH}, OOD_MAX_LENGTH: {OOD_MAX_LENGTH}')

VALID_CHARACTERS = ["s", "a", "b", "e", "p"]
MAIN_CHARACTERS = ["a", "b"]
START_TOKEN = "s"
END_TOKEN = "e"
PADDING_TOKEN = "p"
VALID_RATIO = 0.5  # Half of the dataset should be valid a*b* strings

# Define constants for model
VOCAB_SIZE = len(VALID_CHARACTERS)
EMBEDDING_DIM = 6
NUM_HEADS = 2
NUM_LAYERS = 1
HIDDEN_DIM = 2
BATCH_SIZE = 512
EPOCHS = 15

# Mapping characters to indices
char_to_index = {ch: idx for idx, ch in enumerate(VALID_CHARACTERS)}

# Custom dataset class
class StringDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.labels = []
        with open(file_path, "r") as f:
            for line in f:
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
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

accelerator = Accelerator()

# Prepare dataset and dataloader
dataset = StringDataset("counting_dataset/train_dataset.txt")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss function, and optimizer
model = TransformerClassifier(
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS
)

logger.info(f'''Model hyperparameters: VOCAB_SIZE: {VOCAB_SIZE}, EMBEDDING_DIM: {EMBEDDING_DIM}, 
NUM_HEADS:{NUM_HEADS}, HIDDEN_DIM: {HIDDEN_DIM}, NUM_LAYERS: {NUM_LAYERS}''')

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and torch.backends.mps.is_available():
    device = torch.device("mps")
model.to(device)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

if accelerator.is_main_process:
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
    print(f"Initial Accuracy: {correct/total}")

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        accelerator.backward(loss)
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")
    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

    if accelerator.is_main_process:
        # Evaluate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                outputs = model(inputs)
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()
        print(f"Training Accuracy: {correct/total}")
        logger.info(f"Training Accuracy: {correct/total}")

# Save the trained model
torch.save(model.state_dict(), "simple_test.pth")

test_dataset = StringDataset("counting_dataset/test_dataset.txt")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Print accuracy of the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs.to(device))
        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted.squeeze().to(device) == labels.to(device)).sum().item()
print(f"Test Accuracy: {correct/total}")
logger.info(f"Test Accuracy: {correct/total}")


ood_dataset = StringDataset("counting_dataset/ood_dataset.txt")
ood_dataloader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Print accuracy of the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in ood_dataloader:
        outputs = model(inputs.to(device))
        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted.squeeze().to(device) == labels.to(device)).sum().item()
print(f"OOD Accuracy: {correct/total}")
logger.info(f"OOD Accuracy: {correct/total}")

