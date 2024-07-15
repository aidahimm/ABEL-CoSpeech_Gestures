import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open('data_sequences120.pkl', 'rb') as f:
    data = pickle.load(f)

train_joint_sequences = data['train_joint_sequences']
val_joint_sequences = data['val_joint_sequences']
test_joint_sequences = data['test_joint_sequences']
train_speech_sequences = data['train_speech_sequences']
val_speech_sequences = data['val_speech_sequences']
test_speech_sequences = data['test_speech_sequences']

# Parameters
seq_length = 120
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(TensorDataset(train_speech_sequences, train_joint_sequences), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_speech_sequences, val_joint_sequences), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(test_speech_sequences, test_joint_sequences), batch_size=64, shuffle=False)

# Define the CNN-LSTM-based Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        self.bi_lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers, 
                               batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization on hidden_dim, not seq_length
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.seq_length = seq_length

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # Change to (batch_size, features, seq_length)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        # x = self.relu(self.conv3(x))
        # x = self.pool(x)
        # x = self.dropout(x)
        # x = self.relu(self.conv4(x))
        # x = self.pool(x)
        # x = self.dropout(x)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # Change back to (batch_size, seq_length, features)
        
        lstm_out, _ = self.bi_lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the LSTM
        
        x = self.relu(self.fc1(lstm_out))
        x = self.bn1(x)  # Apply batch normalization on the hidden features
        out = self.fc2(x)
        return self.sigmoid(out).unsqueeze(1).repeat(1, self.seq_length, 1)
    
# Define the CNN-LSTM-based Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        self.bi_lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization on hidden_dim, not seq_length
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # Change to (batch_size, features, seq_length)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        # x = self.relu(self.conv3(x))
        # x = self.pool(x)
        # x = self.dropout(x)
        # x = self.relu(self.conv4(x))
        # x = self.pool(x)
        # x = self.dropout(x)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # Change back to (batch_size, seq_length, features)
        
        lstm_out, _ = self.bi_lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the LSTM
        
        x = self.relu(self.fc1(lstm_out))
        x = self.bn1(x)  # Apply batch normalization on the hidden features
        out = self.fc2(x)
        return self.sigmoid(out).view(-1, 1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_dim = 50
hidden_dim = 128
num_layers = 3
num_epochs = 40
learning_rate_g = 0.00001
learning_rate_d = 0.00004


# Loss function
criterion = nn.BCELoss()

# Model initialization
input_dim = train_speech_sequences.shape[2]
output_dim = train_joint_sequences.shape[2]  # Ensure output_dim matches the joint sequences feature size
netG = Generator(input_dim, hidden_dim, output_dim, num_layers).to(device)
netD = Discriminator(output_dim, hidden_dim, num_layers).to(device)

# Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate_g, weight_decay=1e-5)
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate_d, weight_decay=1e-5)

# Label smoothing values
real_label_smoothing = 0.9
fake_label_smoothing = 0.1

# Initialize metric storage
train_metrics = {'epoch': [], 'Loss_D': [], 'Loss_G': [], 'MAE': [], 'MSE': [], 'RMSE':[], 'R2': []}
val_metrics = {'epoch': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}

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
        real_labels = torch.full((b_size, 1), 0.9, dtype=torch.float, device=device)
        errD_real = criterion(real_output, real_labels)
        errD_real.backward()

        fake_output = netD(fake_joint.detach())
        fake_labels = torch.full((b_size, 1), 0.1, dtype=torch.float, device=device)
        errD_fake = criterion(fake_output, fake_labels)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        fake_output = netD(fake_joint)
        errG = criterion(fake_output, real_labels)
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
        train_joint_sequences_np = train_joint_sequences.cpu().numpy().reshape(-1, output_dim)
        gen_joint_train_np = gen_joint_train.cpu().numpy().reshape(-1, output_dim)
        train_mae = mean_absolute_error(train_joint_sequences_np, gen_joint_train_np)
        train_mse = mean_squared_error(train_joint_sequences_np, gen_joint_train_np)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(train_joint_sequences_np, gen_joint_train_np)

    train_metrics['epoch'].append(epoch)
    train_metrics['Loss_D'].append(errD.item())
    train_metrics['Loss_G'].append(errG.item())
    train_metrics['MAE'].append(train_mae)
    train_metrics['MSE'].append(train_mse)
    train_metrics['RMSE'].append(train_mse)
    train_metrics['R2'].append(train_r2)

    print(f'Train MAE: {train_mae:.4f}',
          f'Train MSE: {train_mse:.4f}',
          f'Train RMSE: {train_rmse:.4f}',
          f'Train R2: {train_r2:.4f}')

    # Evaluate metrics on validation data
    with torch.no_grad():
        gen_joint_val = netG(val_speech_sequences.to(device))
        val_joint_sequences_np = val_joint_sequences.cpu().numpy().reshape(-1, output_dim)
        gen_joint_val_np = gen_joint_val.cpu().numpy().reshape(-1, output_dim)
        val_mae = mean_absolute_error(val_joint_sequences_np, gen_joint_val_np)
        val_mse = mean_squared_error(val_joint_sequences_np, gen_joint_val_np)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(val_joint_sequences_np, gen_joint_val_np)
    val_metrics['epoch'].append(epoch)
    val_metrics['MAE'].append(val_mae)
    val_metrics['MSE'].append(val_mse)
    val_metrics['RMSE'].append(val_rmse)
    val_metrics['R2'].append(val_r2)

    print(f'Val MAE: {val_mae:.4f}', 
        f'Val MSE: {val_mse:.4f}', 
        f'Val RMSE: {val_rmse:.4f}',
        f'Val R2: {val_r2:.4f}')

    # Save checkpoint
    torch.save(netG.state_dict(), f'/Volumes/NO NAME/ABEL-body-motion/cnnGan1_models/generator11.pth')
    torch.save(netD.state_dict(), f'/Volumes/NO NAME/ABEL-body-motion/cnnGan1_models/discriminator11.pth')

metrics_dir = '/Volumes/NO NAME/ABEL-body-motion/cnnGan1_models'
train_metrics_df = pd.DataFrame(train_metrics)
train_metrics_df.to_csv(os.path.join(metrics_dir, 'train_metrics11.csv'), index=False)

val_metrics_df = pd.DataFrame(val_metrics)
val_metrics_df.to_csv(os.path.join(metrics_dir, 'val_metrics11.csv'), index=False)

# Evaluate metrics on test data
with torch.no_grad():
    gen_joint_test = netG(test_speech_sequences.to(device))
    test_joint_sequences_np = test_joint_sequences.cpu().numpy().reshape(-1, output_dim)
    gen_joint_test_np = gen_joint_test.cpu().numpy().reshape(-1, output_dim)
    test_mae = mean_absolute_error(test_joint_sequences_np, gen_joint_test_np)
    test_mse = mean_squared_error(test_joint_sequences_np, gen_joint_test_np)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_joint_sequences_np, gen_joint_test_np)
test_metrics = {
    'MAE': [test_mae],
    'MSE': [test_mse],
    'RMSE': [test_rmse],
    'R2': [test_r2]
}

test_metrics_df = pd.DataFrame(test_metrics)
test_metrics_df.to_csv(os.path.join(metrics_dir, 'test_metrics11.csv'), index=False)

print(f'Test MAE: {test_mae:.4f}',
      f'Test MSE: {test_mse:.4f}',
      f'Test RMSE: {test_rmse:.4f}',
      f'Test R2: {test_r2:.4f}')
