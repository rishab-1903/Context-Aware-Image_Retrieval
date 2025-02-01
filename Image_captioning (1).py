import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import nltk
import pickle
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Paths
DATA_DIR = './flickr30k'
IMAGES_DIR = os.path.join(DATA_DIR, 'Images')
CAPTIONS_FILE = os.path.join(DATA_DIR, 'captions.txt')
MODEL_PATH = './flickr30k/mymodel_30.pth'
VOCAB_PATH = './flickr30k/vocab_30.pkl'

# Hyperparameters
EMBED_SIZE = 512
HIDDEN_SIZE = 768
NUM_LAYERS = 2
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 1
MAX_LENGTH = 40  # Set max_length to 34

# Tokenizer
def build_vocab(caption_file, threshold=5):
    with open(caption_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    counter = Counter()
    for line in lines:
        tokens = nltk.tokenize.word_tokenize(line.split(',')[1].lower())
        counter.update(tokens)
    
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = {word: idx for idx, word in enumerate(words, 4)}
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3
    
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab

# Dataset class
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, caption_file, vocab, transform=None, max_length=40):
        self.img_dir = img_dir
        self.captions = open(caption_file, 'r').readlines()[1:]
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        line = self.captions[idx].strip().split(',')
        img_path = os.path.join(self.img_dir, line[0])
        caption = nltk.tokenize.word_tokenize(line[1].lower())
        
        caption = [self.vocab.get(word, self.vocab['<unk>']) for word in caption]
        caption = [self.vocab['<start>']] + caption + [self.vocab['<end>']]
        
        # Pad or truncate the caption to max_length
        if len(caption) < self.max_length:
            caption = caption + [self.vocab['<pad>']] * (self.max_length - len(caption))
        else:
            caption = caption[:self.max_length]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(caption)

# Data preprocessing
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset with max_length set to 34
vocab = build_vocab(CAPTIONS_FILE)
dataset = Flickr8kDataset(IMAGES_DIR, CAPTIONS_FILE, vocab, transform, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)

    def forward(self, images):
        return self.resnet(images)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions[:, :-1]))
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(inputs)
        return self.fc(lstm_out)

# Initialize models
encoder = EncoderCNN(EMBED_SIZE).to(device)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE, weight_decay = 1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Use gradient scaler for mixed precision training
scaler = GradScaler()

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, captions = images.to(device), captions.to(device)

        optimizer.zero_grad()

        with autocast():  # Mixed precision training
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))

        # Backpropagation with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    # Print epoch loss
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}')

    # 🔹 Step the scheduler here after the epoch is complete
    scheduler.step()

    # 🔹 Print the updated learning rate
    for param_group in optimizer.param_groups:
        print(f"Updated Learning Rate: {param_group['lr']}")

# Save model
torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')

