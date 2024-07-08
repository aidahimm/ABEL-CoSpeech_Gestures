import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.decomposition import PCA

# Define the path to your datasets and processed files
train_dir = '/Volumes/NO NAME/ABEL-body-motion/sets_zipped_features/Train'
val_dir = '/Volumes/NO NAME/ABEL-body-motion/sets_zipped_features/Validation'
test_dir = '/Volumes/NO NAME/ABEL-body-motion/sets_zipped_features/Test'
processed_dir = '/Volumes/NO NAME/ABEL-body-motion/Combined_data'

# Function to identify all unique columns across all files
def get_all_columns(directory):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    all_columns = set()
    for file in all_files:
        df = pd.read_csv(os.path.join(directory, file))
        all_columns.update(df.columns)
    return list(all_columns)

# Function to load data from directory
def load_data_from_directory(directory_path):
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    all_columns = get_all_columns(directory_path)
    dfs = []
    for filename in all_files:
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    combined_df = combined_df.fillna(0)
    combined_df['global_id'] = combined_df['take_id'].astype(str) + '0' + combined_df['frame'].astype(str)
    combined_df = combined_df.reindex(columns=['global_id'] + all_columns, fill_value=0)
    combined_df = combined_df.drop(columns=['frame', 'take_id'])
    for col in combined_df.columns:
        if set(combined_df[col].unique()) == {'True', 'False'}:
            combined_df[col] = combined_df[col].map({'True': 1, 'False': 0})
        elif col == 'global_id':
            combined_df[col] = combined_df[col].astype(int)
        else:
            combined_df[col] = combined_df[col].astype(float)
    return combined_df

def save_dataframe(df, filename):
    filepath = os.path.join(processed_dir, filename)
    df.to_csv(filepath, index=False)

def load_dataframe(filename):
    filepath = os.path.join(processed_dir, filename)
    return pd.read_csv(filepath)

# Check if processed files exist
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

if not os.path.exists(os.path.join(processed_dir, 'train.csv')):
    train_df = load_data_from_directory(train_dir)
    save_dataframe(train_df, 'train.csv')
else:
    print("\nLoading Train...")
    train_df = load_dataframe('train.csv')
    print(train_df.shape)

if not os.path.exists(os.path.join(processed_dir, 'val.csv')):
    val_df = load_data_from_directory(val_dir)
    save_dataframe(val_df, 'val.csv')
else:
    print("\nLoading Validation...")
    val_df = load_dataframe('val.csv')
    print(val_df.shape)

if not os.path.exists(os.path.join(processed_dir, 'test.csv')):
    test_df = load_data_from_directory(test_dir)
    save_dataframe(test_df, 'test.csv')
else:
    print("\nLoading Test...")
    test_df = load_dataframe('test.csv')
    print(test_df.shape)

# Normalize dataframes
def normalize_df(df):
    denominator = df.max() - df.min()
    denominator[denominator == 0] = 1  # Avoid division by zero
    return (df - df.min()) / denominator

train_df = normalize_df(train_df)
val_df = normalize_df(val_df)
test_df = normalize_df(test_df)

# Separate speech and joint position features, including the time column
def separate_features(df):
    speech_features = df.filter(regex='^lemma_|^word_emb_|^time|^tag_|^dep_')
    joint_features = df.drop(columns=[col for col in speech_features.columns if col != 'time'])
    if 'time' not in joint_features.columns:
        joint_features['time'] = df['time']
    joint_features = joint_features.drop(columns=['global_id'])  # Assuming 'global_id' is not a feature
    return speech_features, joint_features

# Separate speech and joint position features, including time
train_speech, train_joint = separate_features(train_df)
val_speech, val_joint = separate_features(val_df)
test_speech, test_joint = separate_features(test_df)

# Apply PCA
def apply_pca(train, val, test, n_components):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train)
    val_pca = pca.transform(val)
    test_pca = pca.transform(test)
    return train_pca, val_pca, test_pca

n_components_speech = 50
n_components_joint = 50

train_speech_pca, val_speech_pca, test_speech_pca = apply_pca(train_speech, val_speech, test_speech, n_components_speech)
train_joint_pca, val_joint_pca, test_joint_pca = apply_pca(train_joint, val_joint, test_joint, n_components_joint)

# Convert to tensors
train_speech_tensor = torch.tensor(train_speech_pca, dtype=torch.float32)
val_speech_tensor = torch.tensor(val_speech_pca, dtype=torch.float32)
test_speech_tensor = torch.tensor(test_speech_pca, dtype=torch.float32)

train_joint_tensor = torch.tensor(train_joint_pca, dtype=torch.float32)
val_joint_tensor = torch.tensor(val_joint_pca, dtype=torch.float32)
test_joint_tensor = torch.tensor(test_joint_pca, dtype=torch.float32)

def create_sequences(data, seq_length):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, seq_length):  # Non-overlapping sequences
        sequences.append(data[i:i+seq_length].unsqueeze(0))  # Add batch dimension
    return torch.cat(sequences, dim=0)

seq_length = 100

train_speech_sequences = create_sequences(train_speech_tensor, seq_length)
val_speech_sequences = create_sequences(val_speech_tensor, seq_length)
test_speech_sequences = create_sequences(test_speech_tensor, seq_length)

train_joint_sequences = create_sequences(train_joint_tensor, seq_length)
val_joint_sequences = create_sequences(val_joint_tensor, seq_length)
test_joint_sequences = create_sequences(test_joint_tensor, seq_length)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(train_speech_sequences, train_joint_sequences), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_speech_sequences, val_joint_sequences), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(test_speech_sequences, test_joint_sequences), batch_size=64, shuffle=False)

print(train_joint_tensor.shape)
print(train_speech_tensor.shape)
number_of_features = train_speech_tensor.shape[1]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Encoder, Decoder, and Seq2Seq classes
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
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

input_dim = train_speech_sequences.shape[2]  # Number of speech features
output_dim = train_joint_sequences.shape[2]  # Number of joint position features
hidden_dim = 256
num_layers = 2

encoder = Encoder(input_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for speech_seq, joint_seq in train_loader:
        speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
        optimizer.zero_grad()
        
        output = model(speech_seq, joint_seq)
        loss = criterion(output, joint_seq)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for speech_seq, joint_seq in val_loader:
            speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
            output = model(speech_seq, joint_seq)
            loss = criterion(output, joint_seq)
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}')

    test_loss = 0
    model.eval()
    with torch.no_grad():
        for speech_seq, joint_seq in test_loader:
            speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
            output = model(speech_seq, joint_seq)
            loss = criterion(output, joint_seq)
            test_loss += loss.item()

    print(f'Test Loss: {test_loss / len(test_loader):.4f}') 

    torch.save(encoder.state_dict(), f"/Volumes/NO NAME/ABEL-body-motion/sgan1_models/encoder_epoch{epoch}.pth")
    torch.save(decoder.state_dict(), f"/Volumes/NO NAME/ABEL-body-motion/sgan1_models/decoder_epoch{epoch}.pth")
    torch.save(model.state_dict(), f"/Volumes/NO NAME/ABEL-body-motion/sgan1_models/model_epoch{epoch}.pth")
