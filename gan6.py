import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# # Load all unique columns from train, validation, and test directories
# all_columns = set(get_all_columns(train_dir))
# all_columns.update(get_all_columns(val_dir))
# all_columns.update(get_all_columns(test_dir))
# all_columns = list(all_columns)

def load_data_from_directory(directory_path):
    dfs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
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
    print(combined_df.head)
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
    joint_features = joint_features.drop(columns=['global_id'] )  # Assuming 'global_id' is not a feature
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

seq_length = 120

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
# Define the LSTM-based Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seq_length):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out)
        return self.sigmoid(out)

# Define the LSTM-based Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return self.sigmoid(out).view(-1, 1)  # Ensure the output is of shape [batch_size, 1]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
hidden_dim = 128
num_layers = 2
output_dim = train_joint_sequences[2]
batch_size = 64
num_epochs = 25
learning_rate_g = 0.001
learning_rate_d = 0.001 
lambda_gp = 10  # Gradient penalty coefficient

# Loss function
criterion = nn.BCELoss()

# Update model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_speech_sequences.shape[2]
netG = Generator(input_dim, hidden_dim, train_joint_sequences.shape[2], num_layers, seq_length).to(device)
netD = Discriminator(train_joint_sequences.shape[2], hidden_dim, num_layers).to(device)

# Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate_g)
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate_d)


# Add gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=5.0)
torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=5.0)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Label smoothing values
real_label = 0.9
fake_label = 0.1

# Gradient penalty computation
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Initialize metric storage
train_metrics = {'epoch': [], 'Loss_D': [], 'Loss_G': [], 'MAE': [], 'RMSE':[], 'R2': []}
val_metrics = {'epoch': [], 'MAE': [], 'RMSE': [], 'R2': []}

# Training Loop
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (speech_data, joint_data) in enumerate(train_loader, 0):
        netD.zero_grad()
        real_joint = joint_data.to(device)
        b_size = real_joint.size(0)
        
        # Generate fake joint position data
        speech_data = speech_data.to(device)
        fake_joint = netG(speech_data)
        
        # Train Discriminator
        real_output = netD(real_joint)
        real_label = torch.full((b_size, 1), 1, dtype=torch.float, device=device)
        errD_real = criterion(real_output, real_label)
        errD_real.backward()
        
        fake_output = netD(fake_joint.detach())
        fake_label = torch.full((b_size, 1), 0, dtype=torch.float, device=device)
        errD_fake = criterion(fake_output, fake_label)
        errD_fake.backward()
        
        gradient_penalty = compute_gradient_penalty(netD, real_joint.data, fake_joint.data)
        errD = errD_real + errD_fake + lambda_gp * gradient_penalty
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        fake_output = netD(fake_joint)
        errG = criterion(fake_output, real_label)
        errG.backward()
        optimizerG.step()

        # Print progress
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {real_output.mean().item():.4f} D(G(z)): {fake_output.mean().item():.4f}')

    # Evaluate metrics on training data
    with torch.no_grad():
        gen_joint_train = netG(train_speech_sequences.to(device))
        train_joint_sequences_flat = train_joint_sequences.view(-1, train_joint_sequences.shape[-1]).cpu().numpy()
        gen_joint_train_flat = gen_joint_train.view(-1, gen_joint_train.shape[-1]).cpu().numpy()
        train_mae = mean_absolute_error(train_joint_sequences_flat, gen_joint_train_flat)
        train_mse = mean_squared_error(train_joint_sequences_flat, gen_joint_train_flat)
        val_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(train_joint_sequences_flat, gen_joint_train_flat)

    train_metrics['epoch'].append(epoch)
    train_metrics['Loss_D'].append(errD.item())
    train_metrics['Loss_G'].append(errG.item())
    train_metrics['MAE'].append(train_mae)
    train_metrics['RMSE'].append(train_mse)
    train_metrics['R2'].append(train_r2)

    # Evaluate metrics on validation data
    with torch.no_grad():
        gen_joint_val = netG(val_speech_sequences.to(device))
        val_joint_sequences_flat = val_joint_sequences.view(-1, val_joint_sequences.shape[-1]).cpu().numpy()
        gen_joint_val_flat = gen_joint_val.view(-1, gen_joint_val.shape[-1]).cpu().numpy()
        val_mae = mean_absolute_error(val_joint_sequences_flat, gen_joint_val_flat)
        val_mse = mean_squared_error(val_joint_sequences_flat, gen_joint_val_flat)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(val_joint_sequences_flat, gen_joint_val_flat)
    val_metrics['epoch'].append(epoch)
    val_metrics['MAE'].append(val_mae)
    val_metrics['RMSE'].append(val_rmse)
    val_metrics['R2'].append(val_r2)

    print(f'Val MAE: {val_mae:.4f}')
    print(f'Val RMSE: {val_rmse:.4f}')
    print(f'Val R2: {val_r2:.4f}')

    # Save checkpoint
    torch.save(netG.state_dict(), f'/Volumes/NO NAME/ABEL-body-motion/gan6_models/generator.pth')
    torch.save(netD.state_dict(), f'/Volumes/NO NAME/ABEL-body-motion/gan6_models/discriminator.pth')

metrics_dir = '/Volumes/NO NAME/ABEL-body-motion/gan6_models'
train_metrics_df = pd.DataFrame(train_metrics)
train_metrics_df.to_csv(os.path.join(metrics_dir, 'train_metrics.csv'), index=False)

val_metrics_df = pd.DataFrame(val_metrics)
val_metrics_df.to_csv(os.path.join(metrics_dir, 'val_metrics.csv'), index=False)

# Evaluate metrics on test data
with torch.no_grad():
    gen_joint_test = netG(test_speech_sequences.to(device))
    test_joint_sequences_flat = test_joint_sequences.view(-1, test_joint_sequences.shape[-1]).cpu().numpy()
    gen_joint_test_flat = gen_joint_test.view(-1, gen_joint_test.shape[-1]).cpu().numpy()
    test_mae = mean_absolute_error(test_joint_sequences_flat, gen_joint_test_flat)
    test_mse = mean_squared_error(test_joint_sequences_flat, gen_joint_test_flat)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_joint_sequences_flat, gen_joint_test_flat)

test_metrics = {
    'MAE': [test_mae],
    'RMSE': [test_rmse],
    'R2': [test_r2]
}

test_metrics_df = pd.DataFrame(test_metrics)
test_metrics_df.to_csv(os.path.join(metrics_dir, 'test_metrics.csv'), index=False)

print(f'Test MAE: {test_mae:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test R2: {test_r2:.4f}')
