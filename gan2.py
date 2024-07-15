import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open('data_sequences.pkl', 'rb') as f:
    data = pickle.load(f)

train_joint_sequences = data['train_joint_sequences']
val_joint_sequences = data['val_joint_sequences']
test_joint_sequences = data['test_joint_sequences']
train_speech_sequences = data['train_speech_sequences']
val_speech_sequences = data['val_speech_sequences']
test_speech_sequences = data['test_speech_sequences']

seq_length = 150

# Create DataLoaders
train_loader = DataLoader(TensorDataset(train_speech_sequences, train_joint_sequences), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_speech_sequences, val_joint_sequences), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(test_speech_sequences, test_joint_sequences), batch_size=64, shuffle=False)

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
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim_d)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return self.sigmoid(out).view(-1, 1)  # Ensure the output is of shape [batch_size, 1]

# Hyperparameters
hidden_dim = 128
num_layers = 2
output_dim_d = 1
batch_size = 64
num_epochs = 25
learning_rate_g = 0.00005  # Lower learning rate for Generator
learning_rate_d = 0.0004  # Slightly higher learning rate for Discriminator
# lambda_gp = 10  # Gradient penalty coefficient

# Loss function
criterion = nn.BCELoss()

# Update model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_speech_sequences.shape[2]
netG = Generator(input_dim, hidden_dim, train_joint_sequences.shape[2], num_layers, seq_length).to(device)
netD = Discriminator(train_joint_sequences.shape[2], hidden_dim, num_layers).to(device)

# Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate_g, weight_decay=1e-5)
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate_d, weight_decay=1e-5)


# Add gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)


schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.1)
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=10, gamma=0.1)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Label smoothing values
real_label = 0.9
fake_label = 0.1

# # Gradient penalty computation
# def compute_gradient_penalty(D, real_samples, fake_samples):
#     alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
#     interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
#     d_interpolates = D(interpolates)
#     fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.reshape(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

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
        
        # gradient_penalty = compute_gradient_penalty(netD, real_joint.data, fake_joint.data)
        # errD = errD_real + errD_fake + lambda_gp * gradient_penalty
        # optimizerD.step()

        errD = errD_real + errD_fake 
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

    schedulerG.step()
    schedulerD.step()

    # Evaluate metrics on training data
    with torch.no_grad():
        gen_joint_train = netG(train_speech_sequences.to(device))
        train_joint_sequences_flat = train_joint_sequences.view(-1, train_joint_sequences.shape[-1]).cpu().numpy()
        gen_joint_train_flat = gen_joint_train.view(-1, gen_joint_train.shape[-1]).cpu().numpy()
        train_mae = mean_absolute_error(train_joint_sequences_flat, gen_joint_train_flat)
        train_mse = mean_squared_error(train_joint_sequences_flat, gen_joint_train_flat)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(train_joint_sequences_flat, gen_joint_train_flat)

    train_metrics['epoch'].append(epoch)
    train_metrics['Loss_D'].append(errD.item())
    train_metrics['Loss_G'].append(errG.item())
    train_metrics['MAE'].append(train_mae)
    train_metrics['MSE'].append(train_mse)
    train_metrics['RMSE'].append(train_rmse)
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
    val_metrics['MSE'].append(val_mse)
    val_metrics['RMSE'].append(val_rmse)
    val_metrics['R2'].append(val_r2)

    print(f'Val MAE: {val_mae:.4f}')
    print(f'Val RMSE: {val_rmse:.4f}')
    print(f'Val R2: {val_r2:.4f}')

    # Save checkpoint
    torch.save(netG.state_dict(), f'/Volumes/NO NAME/ABEL-body-motion/gan2_models/generator2.pth')
    torch.save(netD.state_dict(), f'/Volumes/NO NAME/ABEL-body-motion/gan2_models/discriminator2.pth')

metrics_dir = '/Volumes/NO NAME/ABEL-body-motion/gan2_models'
train_metrics_df = pd.DataFrame(train_metrics)
train_metrics_df.to_csv(os.path.join(metrics_dir, 'train_metrics2.csv'), index=False)

val_metrics_df = pd.DataFrame(val_metrics)
val_metrics_df.to_csv(os.path.join(metrics_dir, 'val_metrics2.csv'), index=False)

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
test_metrics_df.to_csv(os.path.join(metrics_dir, 'test_metrics2.csv'), index=False)

print(f'Test MAE: {test_mae:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test R2: {test_r2:.4f}')


