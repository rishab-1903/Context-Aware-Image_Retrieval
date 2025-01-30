import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os

# Paths to the model and vocabulary
MODEL_PATH = 'mymodel (2).pth'
VOCAB_PATH = 'vocab (1).pkl'

# Hyperparameters (same as training)
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
MAX_LENGTH = 34  # Set max_length used during training

# Load vocabulary
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

# Reverse vocabulary mapping
idx_to_word = {idx: word for word, idx in vocab.items()}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Encoder and Decoder models
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)

    def forward(self, images):
        return self.resnet(images)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def generate_caption(self, features, max_length):
        generated_caption = []
        input_word = torch.tensor([vocab['<start>']]).unsqueeze(0).to(device)

        for _ in range(max_length):
            embedding = self.embedding(input_word)
            lstm_out, _ = self.lstm(embedding)
            output = self.fc(lstm_out.squeeze(1))
            predicted_idx = output.argmax(1).item()
            generated_caption.append(predicted_idx)

            if predicted_idx == vocab['<end>']:
                break
            input_word = torch.tensor([predicted_idx]).unsqueeze(0).to(device)

        return [idx_to_word[idx] for idx in generated_caption]

# Load models
encoder = EncoderCNN(EMBED_SIZE).to(device)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()

# Image preprocessing
# Define the transformation pipeline (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def generate_caption(input_image):
    """
    Generate a caption for an input image.

    Args:
        input_image: Can be a file path (str) or a PIL.Image object.

    Returns:
        str: Generated caption for the input image.
    """
    # Check if the input is a file path or a PIL.Image
    if isinstance(input_image, str):
        # If it's a file path, open the image
        image = Image.open(input_image).convert('RGB')
    elif isinstance(input_image, Image.Image):
        # If it's already a PIL.Image, use it directly
        image = input_image
    else:
        raise ValueError("Input must be a file path or a PIL.Image object")

    # Apply transformations and prepare for the model
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        features = encoder(image_tensor)
        caption_tokens = decoder.generate_caption(features, MAX_LENGTH)

    # Convert tokens to words, and remove <start> and <end> tokens
    caption = [word for word in caption_tokens if word not in ('<start>', '<end>')]
    return ' '.join(caption)
