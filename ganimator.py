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

def apply_pca(train_df, val_df, test_df, n_components=100):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_df)
    val_pca = pca.transform(val_df)
    test_pca = pca.transform(test_df)
    return train_pca, val_pca, test_pca

# Apply PCA to your dataframes
n_components = 100  # Number of principal components
train_pca, val_pca, test_pca = apply_pca(train_df, val_df, test_df, n_components)

# Convert PCA data to tensors
train_tensor = torch.tensor(train_pca, dtype=torch.float32)
val_tensor = torch.tensor(val_pca, dtype=torch.float32)
test_tensor = torch.tensor(test_pca, dtype=torch.float32)

# # Convert dataframes to tensors
# def dataframe_to_tensor(df):
#     return torch.tensor(df.values, dtype=torch.float32)

# train_tensor = dataframe_to_tensor(train_df)
# val_tensor = dataframe_to_tensor(val_df)
# test_tensor = dataframe_to_tensor(test_df)

def create_sequences(data, seq_length):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, seq_length):  # Non-overlapping sequences
        sequences.append(data[i:i+seq_length].unsqueeze(0))  # Add batch dimension
    return torch.cat(sequences, dim=0)

seq_length = 100

train_sequences = create_sequences(train_tensor, seq_length)
val_sequences = create_sequences(val_tensor, seq_length)
test_sequences = create_sequences(test_tensor, seq_length)

# Update your DataLoaders
train_loader = DataLoader(TensorDataset(train_sequences), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_sequences), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(test_sequences), batch_size=64, shuffle=False)

print(train_tensor.shape)
number_of_features = train_tensor.shape[1]

# Define the LSTM-based Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seq_length):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
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
output_dim = 1
batch_size = 64
num_epochs = 25
learning_rate_g = 0.0001  # Lower learning rate for Generator
learning_rate_d = 0.0004  # Slightly higher learning rate for Discriminator
lambda_gp = 10  # Gradient penalty coefficient

# Loss function
criterion = nn.BCELoss()

# Update model initialization
input_dim = train_sequences.shape[2]  # Number of features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(input_dim, hidden_dim, input_dim, num_layers, seq_length).to(device)
netD = Discriminator(input_dim, hidden_dim, output_dim, num_layers).to(device)

# Add gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Label smoothing values
real_label = 0.9
fake_label = 0.1

# Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate_g)
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate_d)

# Gradient penalty computation
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(device)  # Changed to 3 dimensions
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)  # Ensure the shape matches
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)  # Use reshape instead of view
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training Loop

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        # Update D: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        # Train with real data
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)  # Change shape to [b_size, 1]
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake data
        noise = torch.randn(b_size, seq_length, input_dim, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(netD, real_cpu.data, fake.data)
        
        # Total Discriminator loss
        errD = errD_real + errD_fake + lambda_gp * gradient_penalty
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        label.fill_(real_label)  # Generator wants discriminator to think samples are real
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Print progress
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'generator2.pth': netG.state_dict(),
        'discriminator2.pth': netD.state_dict(),
        'optimizerG_2_state_dict': optimizerG.state_dict(),
        'optimizerD_2_state_dict': optimizerD.state_dict(),
    }, f'/Volumes/NO NAME/ABEL-body-motion/ganimator2/checkpoint_2_epoch_{epoch}.pth')
