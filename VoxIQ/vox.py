import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

#torch.manual_seed(0)
#np.random.seed(0)
#random.seed(0)
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# Define a simple classification network
class SpeechClassifier(nn.Module):
    def __init__(self, num_classes=4, hidden_size=128):
        super(SpeechClassifier, self).__init__()

        self.fc1 = nn.Linear(13 * 217, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class SpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
        if not self.audio_files:
            raise ValueError("No audio files found in the specified directory.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.audio_files[idx])
        audio_data, _ = librosa.load(file_path, sr=16000)
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
        mfcc_features_flattened = mfcc_features.flatten()

        max_length = 2821
        if len(mfcc_features_flattened) < max_length:
            mfcc_features_padded = np.pad(mfcc_features_flattened, (0, max_length - len(mfcc_features_flattened)))
        elif len(mfcc_features_flattened) > max_length:
            mfcc_features_padded = mfcc_features_flattened[:max_length]
        else:
            mfcc_features_padded = mfcc_features_flattened

        label = torch.LongTensor([0])  # Make sure this matches the expected dimension
        if label.nelement() == 0:
            raise ValueError("Label tensor is empty")

        return torch.FloatTensor(mfcc_features_padded), label



# Function to preprocess user audio data with fixed length MFCC features
def preprocess_user_audio(audio_file_path, max_length=2821):
    # Load and preprocess the user audio file
    audio_data, _ = librosa.load(audio_file_path, sr=16000)  # Adjust sample rate as needed
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)  # Adjust parameters as needed

    # Flatten the MFCC features to match the expected input size of the model
    mfcc_features_flattened = mfcc_features.flatten()
    
    # Pad or truncate the flattened MFCC features to match the maximum length
    if len(mfcc_features_flattened) < max_length:
        mfcc_features_padded = np.pad(mfcc_features_flattened, (0, max_length - len(mfcc_features_flattened)))
    elif len(mfcc_features_flattened) > max_length:
        mfcc_features_padded = mfcc_features_flattened[:max_length]
    else:
        mfcc_features_padded = mfcc_features_flattened

    # Convert the padded MFCC features to a PyTorch tensor
    input_tensor = torch.FloatTensor(mfcc_features_padded)

    # Return the preprocessed audio data
    return input_tensor



# Function to classify user audio using the trained model
def classify_user_audio(audio_file_path, model):
    # Preprocess the user audio data
    input_data = preprocess_user_audio(audio_file_path)

    # Pass the preprocessed data through the model to get predictions
    with torch.no_grad():
        output = model(input_data.unsqueeze(0))  # Add an extra dimension for batch size

    # Perform post-processing based on your task (classification or similarity)
    # For classification, return the class with the highest probability
    predicted_class = torch.argmax(output).item()

    return predicted_class

# Specify the root folder containing the dataset
root_folder = '/Users/bakhodirulugov/Desktop/Backend/Dataset'

# Create dataset and dataloader
dataset = SpeechDataset(root_folder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


# Initialize the model, loss function, and optimizer
model = SpeechClassifier(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for classification
epochs = 10
for epoch in range(epochs):
    for inputs, labels in dataloader:
        print(f'Batch Inputs size: {inputs.size()}, Batch Labels size: {labels.size()}')
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels.squeeze())
        loss.backward()
        optimizer.step()


# Specify the path to the user audio file
user_audio_file = '/Users/bakhodirulugov/Desktop/Backend/User Audio/84_121123_000008_000000.wav'

# Classify user audio using the trained model
predicted_class = classify_user_audio(user_audio_file, model)

# Print or visualize the result
print(f'Predicted Class for User Audio: {predicted_class}')

# Initialize the model
model = SpeechClassifier(num_classes=4)
torch.save(model.state_dict(), 'model.pth')


