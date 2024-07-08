import os 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
import joblib
import json

processed_dir = "/Volumes/NO NAME/ABEL-body-motion/Combined_data"

def load_dataframe(filename):
    filepath = os.path.join(processed_dir, filename)
    return pd.read_csv(filepath)

print("\nLoading Test...")
test_df = load_dataframe('test.csv')
print(test_df.shape)

# Load min and max values from the saved JSON file
def load_min_max(filename):
    with open(filename, 'r') as f:
        min_max = json.load(f)
    min_vals = pd.Series(min_max['min'])
    max_vals = pd.Series(min_max['max'])
    return min_vals, max_vals

train_min, train_max = load_min_max('train_min_max.json')

# Normalize dataframes using the training set statistics
def normalize_df(df, min_vals, max_vals):
    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1  # Avoid division by zero
    return (df - min_vals) / denominator

test_df = normalize_df(test_df, train_min, train_max)

# Separate speech and joint position features, including the time column
def separate_features(df):
    speech_features = df.filter(regex='^lemma_|^word_emb_|^time|^tag_|^dep_')
    joint_features = df.drop(columns=[col for col in speech_features.columns if col != 'time'])
    if 'time' not in joint_features.columns:
        joint_features['time'] = df['time']
    joint_features = joint_features.drop(columns=['global_id'])  # Assuming 'global_id' is not a feature
    return speech_features, joint_features

test_speech, test_joint = separate_features(test_df)
original_joint_columns = test_joint.columns.tolist()
original_speech_columns = test_speech.columns.tolist()

subset_test_speech = test_speech[:49574]

# Define the path to the saved model and PCA models
model_path = '/Volumes/NO NAME/ABEL-body-motion/tgan2_models/best_model2.pth'
pca_speech_path = '/Volumes/NO NAME/ABEL-body-motion/pca_speech_model.pkl'
pca_joint_path = '/Volumes/NO NAME/ABEL-body-motion/pca_joint_model.pkl'

# Load the model classes
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs, _, _ = self.decoder(trg, hidden, cell)
        return outputs

# Assume the same model parameters used during training
input_dim = 50  # Adjust based on your PCA components
output_dim = 50  # Adjust based on your PCA components
hidden_dim = 256
num_layers = 2

# Initialize the model
encoder = Encoder(input_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder)

# Load the trained model parameters
model.load_state_dict(torch.load(model_path))
model.eval()

# Determine the device and move the model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the PCA models
pca_speech = joblib.load(pca_speech_path)
pca_joint = joblib.load(pca_joint_path)

# Apply the loaded PCA models to the test data
subset_test_speech_pca = pca_speech.transform(subset_test_speech)

# Convert to tensors
subset_test_speech_tensor = torch.tensor(subset_test_speech_pca, dtype=torch.float32)

# Create sequences
seq_length = 100
def create_sequences(data, seq_length):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, seq_length):  # Non-overlapping sequences
        sequences.append(data[i:i+seq_length].unsqueeze(0))  # Add batch dimension
    return torch.cat(sequences, dim=0)

subset_test_speech_sequences = create_sequences(subset_test_speech_tensor, seq_length)

# Create DataLoader for the test speech data
test_loader = DataLoader(TensorDataset(subset_test_speech_sequences), batch_size=64, shuffle=False)

# Predict the joint data
predictions = []
with torch.no_grad():
    for speech_seq in test_loader:
        speech_seq = speech_seq[0].to(device)
        output = model(speech_seq, speech_seq)
        predictions.append(output.cpu().numpy())

# Concatenate all predictions
predictions = np.concatenate(predictions, axis=0)

# Reshape predictions to 2D
predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])

# Reverse PCA transformation
predictions_original = pca_joint.inverse_transform(predictions_reshaped)

def unnormalize_df(df, min_vals, max_vals):
    denominator = max_vals - min_vals
    unnormalized_df = (df * denominator) + min_vals
    return unnormalized_df

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions_original, columns=original_joint_columns)

# Unnormalize the predicted joint data
unnormalized_predictions_df = unnormalize_df(predictions_df, train_min[original_joint_columns], train_max[original_joint_columns])

# Save predictions to a CSV file
output_file = 'predicted_joint_positions.csv'
unnormalized_predictions_df.to_csv(output_file, index=False)


print(f'Predictions saved to {output_file}')