import os
import pickle
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
with open('data_sequences120.pkl', 'rb') as f:
    data = pickle.load(f)

train_joint_sequences = data['train_joint_sequences']
val_joint_sequences = data['val_joint_sequences']
test_joint_sequences = data['test_joint_sequences']
train_speech_sequences = data['train_speech_sequences']
val_speech_sequences = data['val_speech_sequences']
test_speech_sequences = data['test_speech_sequences']

seq_length = 120

# Create DataLoaders
train_loader = DataLoader(TensorDataset(train_speech_sequences, train_joint_sequences), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_speech_sequences, val_joint_sequences), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(test_speech_sequences, test_joint_sequences), batch_size=64, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Encoder, Decoder, Seq2Seq, and Discriminator classes
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

    def forward(self, source, target):
        hidden, cell = self.encoder(source)
        outputs, hidden, cell = self.decoder(target, hidden, cell)
        return outputs

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        outputs = outputs[:, -1, :]  # Use the output of the last time step
        validity = self.fc(outputs)
        return validity


# Initialize models
input_dim = train_speech_sequences.shape[-1]
output_dim = train_joint_sequences.shape[-1]
hidden_dim = 128
num_layers = 2

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

encoder = Encoder(input_dim, hidden_dim, num_layers).to(device)
decoder = Decoder(output_dim, hidden_dim, num_layers).to(device)
generator = Seq2Seq(encoder, decoder).to(device)
discriminator = Discriminator(output_dim, hidden_dim, num_layers).to(device)

# Loss functions and optimizers
adversarial_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()

generator.apply(weights_init)
discriminator.apply(weights_init)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

# Training
num_epochs = 40
# early_stopping_patience = 10

best_val_loss = float('inf')
# epochs_no_improve = 0

train_metrics = {'epoch': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}
val_metrics = {'epoch': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}

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

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    train_g_loss = 0
    train_d_loss = 0
    for i, (speech_seq, joint_seq) in enumerate(train_loader, 0):
        speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)

        valid = torch.ones(speech_seq.size(0), 1).to(device)
        fake = torch.zeros(speech_seq.size(0), 1).to(device)

        #  Train Generator
        optimizer_G.zero_grad()

        gen_joint_seq = generator(speech_seq, joint_seq)

        g_adv_loss = adversarial_loss(discriminator(gen_joint_seq), valid)
        g_mse_loss = mse_loss(gen_joint_seq, joint_seq)
        g_loss = g_adv_loss + g_mse_loss
        g_loss.backward()
        optimizer_G.step()

        #  Train Discriminator
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(joint_seq), valid)
        fake_loss = adversarial_loss(discriminator(gen_joint_seq.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        gradient_penalty = compute_gradient_penalty(discriminator, joint_seq.data, gen_joint_seq.data)
        d_loss += 10 * gradient_penalty  # Add gradient penalty
        d_loss.backward()
        optimizer_D.step()

        train_g_loss += g_loss.item()
        train_d_loss += d_loss.item()

    train_g_loss /= len(train_loader)
    train_d_loss /= len(train_loader)

    # Evaluate metrics on training data
    with torch.no_grad():
        gen_joint_train = generator(train_speech_sequences.to(device), train_joint_sequences.to(device))
        train_joint_sequences_flat = train_joint_sequences.view(-1, train_joint_sequences.shape[-1]).cpu().numpy()
        gen_joint_train_flat = gen_joint_train.view(-1, gen_joint_train.shape[-1]).cpu().numpy()
        train_mae = mean_absolute_error(train_joint_sequences_flat, gen_joint_train_flat)
        train_mse = mean_squared_error(train_joint_sequences_flat, gen_joint_train_flat)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(train_joint_sequences_flat, gen_joint_train_flat)

    train_metrics['epoch'].append(epoch)
    train_metrics['MAE'].append(train_mae)
    train_metrics['MSE'].append(train_mse)
    train_metrics['RMSE'].append(train_rmse)
    train_metrics['R2'].append(train_r2)


    # Validation
    generator.eval()
    val_loss = 0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for speech_seq, joint_seq in val_loader:
            speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
            gen_joint_seq = generator(speech_seq, joint_seq)
            val_loss += mse_loss(gen_joint_seq, joint_seq).item()
            y_true_val.append(joint_seq.cpu().numpy())
            y_pred_val.append(gen_joint_seq.cpu().numpy())

    val_loss /= len(val_loader)

    y_true_val = np.concatenate(y_true_val, axis=0).reshape(-1, output_dim)
    y_pred_val = np.concatenate(y_pred_val, axis=0).reshape(-1, output_dim)

    val_mae = mean_absolute_error(y_true_val, y_pred_val)
    val_mse = mean_squared_error(y_true_val, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_true_val, y_pred_val)

    val_metrics['epoch'].append(epoch + 1)
    val_metrics['MAE'].append(val_mae)
    val_metrics['MSE'].append(val_mse)
    val_metrics['RMSE'].append(val_rmse)
    val_metrics['R2'].append(val_r2)

    print(f'Epoch {epoch+1},'
          f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}]'
          f'Loss_D: {train_d_loss} Loss_G: {train_g_loss}'
          f'Training MAE: {train_mae:.4f}, Training MSE: {train_mse:.4f}, Training RMSE: {train_rmse:.4f}, Training R2: {train_r2:.4f},'
          f'Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f},  Validation RMSE: {val_rmse:.4f}, Validation R2: {val_r2:.4f}')


#     # Early stopping
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         epochs_no_improve = 0
#         torch.save(generator.state_dict(), f"/Volumes/NO NAME/ABEL-body-motion/cgan1_models/best_generator.pth")
#         torch.save(discriminator.state_dict(), f"/Volumes/NO NAME/ABEL-body-motion/cgan1_models/best_discriminator.pth")
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve == early_stopping_patience:
#             print("Early stopping")
#             break


metrics_dir = '/Volumes/NO NAME/ABEL-body-motion/cgan1_models'
os.makedirs(metrics_dir, exist_ok=True)
train_metrics_df = pd.DataFrame(train_metrics)
train_metrics_df.to_csv(os.path.join(metrics_dir, 'train_metrics.csv'), index=False)

val_metrics_df = pd.DataFrame(val_metrics)
val_metrics_df.to_csv(os.path.join(metrics_dir, 'val_metrics.csv'), index=False)

# Testing
test_loss = 0
generator.eval()
y_true_test = []
y_pred_test = []
with torch.no_grad():
    for speech_seq, joint_seq in test_loader:
        speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
        gen_joint_seq = generator(speech_seq, joint_seq)
        test_loss += mse_loss(gen_joint_seq, joint_seq).item()
        y_true_test.append(joint_seq.cpu().numpy())
        y_pred_test.append(gen_joint_seq.cpu().numpy())

test_loss /= len(test_loader)

y_true_test = np.concatenate(y_true_test, axis=0).reshape(-1, output_dim)
y_pred_test = np.concatenate(y_pred_test, axis=0).reshape(-1, output_dim)

test_mae = mean_absolute_error(y_true_test, y_pred_test)
test_mse = mean_squared_error(y_true_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_true_test, y_pred_test)

test_metrics = {
    'MAE': [test_mae],
    'MSE': [test_mse],
    'RMSE': [test_rmse],
    'R2': [test_r2]
}

test_metrics_df = pd.DataFrame(test_metrics)
test_metrics_df.to_csv(os.path.join(metrics_dir, 'test_metrics.csv'), index=False)

print(f'Test MAE: {test_mae:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test R2: {test_r2:.4f}')
