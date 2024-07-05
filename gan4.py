import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.express as px

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

seq_length = 150

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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out)
        return self.sigmoid(out)
    
def evaluate_discriminator(discriminator, val_loader):
    discriminator.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            val_real_cpu = data[0].to(device)
            val_output = discriminator(val_real_cpu)
            val_loss += torch.mean(val_output).item()
    discriminator.train()
    return val_loss / len(val_loader)
    
# Loss functions for WGAN-GP
def discriminator_loss(real_output, fake_output, gradient_penalty, lambda_gp):
    return torch.mean(fake_output) - torch.mean(real_output) + lambda_gp * gradient_penalty

def generator_loss(fake_output):
    return -torch.mean(fake_output)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1, :])
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
hidden_dim = 128
num_layers = 2
output_dim = 1
batch_size = 64
num_epochs = 25
learning_rate_g = 0.00001  # Lower learning rate for Generator
learning_rate_d = 0.0001  # Slightly higher learning rate for Discriminator
lambda_gp = 5  # Gradient penalty coefficient

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

# Loss function
criterion = nn.BCELoss()

# Update model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_speech_sequences.shape[2]
netG = Generator(input_dim, hidden_dim, output_dim, num_layers, seq_length).to(device)
netD = Discriminator(output_dim, hidden_dim, 1, num_layers).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

# Add gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Label smoothing values
real_label = 0.9
fake_label = 0.1

# Optimizers
optimizerG = optim.RMSprop(netG.parameters(), lr=learning_rate_g)
optimizerD = optim.RMSprop(netD.parameters(), lr=learning_rate_d)

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

# Initialize lists to store metrics
epochs_list = []
losses_d = []
losses_g = []
wasserstein_distances = []

# Training Loop with Wasserstein distance logging
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (speech_data, joint_data) in enumerate(train_loader, 0):
        netD.zero_grad()
        real_joint = joint_data.to(device)
        b_size = real_joint.size(0)
        noise = torch.randn(b_size, seq_length, input_dim, device=device)
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

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(netD, real_joint.data, fake_joint.data)
        errD = errD_real + errD_fake + lambda_gp * gradient_penalty
        optimizerD.step()
        
        # Train Generator
        netG.zero_grad()
        fake_output = netD(fake_joint)
        errG = criterion(fake_output, real_label)
        errG.backward()
        optimizerG.step()

        # Log Wasserstein distance and other metrics
        if i % 50 == 0:
            wasserstein_distance = (torch.mean(real_output) - torch.mean(fake_output)).item()
            print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'Wasserstein_D: {wasserstein_distance:.4f}')
            epochs_list.append(epoch + i / len(train_loader))
            losses_d.append(errD.item())
            losses_g.append(errG.item())
            wasserstein_distances.append(wasserstein_distance)
            
    # Periodically evaluate discriminator on validation set
    if epoch % 5 == 0:
        val_loss = evaluate_discriminator(netD, val_loader)
        print(f'Epoch {epoch} Validation Loss: {val_loss:.4f}')

    # Save checkpoint
    torch.save(netG.state_dict(), "/Volumes/NO NAME/ABEL-body-motion/gan4_models/generator4.pth")
    torch.save(netD.state_dict(), "/Volumes/NO NAME/ABEL-body-motion/gan4_models/discriminator4.pth")

# Create a DataFrame to store metrics
df_metrics = pd.DataFrame({
    'Epoch': epochs_list,
    'Discriminator Loss': losses_d,
    'Generator Loss': losses_g,
    'Wasserstein Distance': wasserstein_distances
})


# Plotting with Plotly
fig = px.line(df_metrics, x='Epoch', y=['Discriminator Loss', 'Generator Loss', 'Wasserstein Distance'],
              title='Training Metrics Over Epochs')
fig.update_layout(
    xaxis_title='Epoch',
    yaxis_title='Loss',
    legend_title='Metrics',
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
fig.show()
