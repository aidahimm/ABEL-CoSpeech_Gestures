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
train_loader = DataLoader(TensorDataset(train_speech_sequences, train_joint_sequences), batch_size=8, shuffle=True)
val_loader = DataLoader(TensorDataset(val_speech_sequences, val_joint_sequences), batch_size=8, shuffle=False)
test_loader = DataLoader(TensorDataset(test_speech_sequences, test_joint_sequences), batch_size=8, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Encoder, Decoder, and Seq2Seq classes
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        # outputs = self.dropout(outputs)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        # outputs = self.dropout(outputs)  
        predictions = self.fc(outputs)
        # predictions = self.dropout(predictions)
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
hidden_dim = 128
num_layers = 2
dropout = 0.4

encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
decoder = Decoder(output_dim, hidden_dim, num_layers, dropout)
model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 40
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

train_metrics = {'epoch': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}
val_metrics = {'epoch': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    y_true_train = []
    y_pred_train = []
    for speech_seq, joint_seq in train_loader:
        speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
        optimizer.zero_grad()
        
        output = model(speech_seq, joint_seq)
        loss = criterion(output, joint_seq)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        y_true_train.append(joint_seq.detach().cpu().numpy())
        y_pred_train.append(output.detach().cpu().numpy())

    epoch_loss /= len(train_loader)

    y_true_train = np.concatenate(y_true_train, axis=0).reshape(-1, output_dim)
    y_pred_train = np.concatenate(y_pred_train, axis=0).reshape(-1, output_dim)
    
    train_mae = mean_absolute_error(y_true_train, y_pred_train)
    train_mse = mean_squared_error(y_true_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_true_train, y_pred_train)

    train_metrics['epoch'].append(epoch + 1)
    train_metrics['MAE'].append(train_mae)
    train_metrics['MSE'].append(train_mse)
    train_metrics['RMSE'].append(train_rmse)
    train_metrics['R2'].append(train_r2)
    
    val_loss = 0
    model.eval()
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for speech_seq, joint_seq in val_loader:
            speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
            output = model(speech_seq, joint_seq)
            loss = criterion(output, joint_seq)
            val_loss += loss.item()
            y_true_val.append(joint_seq.cpu().numpy())
            y_pred_val.append(output.cpu().numpy())
    
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

    print(f'Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, '
          f'Training MAE: {train_mae:.4f}, Training RMSE: {train_rmse:.4f}, Training R2: {train_r2:.4f}, '
          f'Validation MAE: {val_mae:.4f}, Validation RMSE: {val_rmse:.4f}, Validation R2: {val_r2:.4f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"/Volumes/NO NAME/ABEL-body-motion/tgan2_models/best_model2222.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_patience:
            print("Early stopping")
            break

metrics_dir = '/Volumes/NO NAME/ABEL-body-motion/tgan2_models'
os.makedirs(metrics_dir, exist_ok=True)
train_metrics_df = pd.DataFrame(train_metrics)
train_metrics_df.to_csv(os.path.join(metrics_dir, 'train_metrics2222.csv'), index=False)

val_metrics_df = pd.DataFrame(val_metrics)
val_metrics_df.to_csv(os.path.join(metrics_dir, 'val_metrics2222.csv'), index=False)

test_loss = 0
model.eval()
y_true_test = []
y_pred_test = []
with torch.no_grad():
    for speech_seq, joint_seq in test_loader:
        speech_seq, joint_seq = speech_seq.to(device), joint_seq.to(device)
        output = model(speech_seq, joint_seq)
        loss = criterion(output, joint_seq)
        test_loss += loss.item()
        y_true_test.append(joint_seq.cpu().numpy())
        y_pred_test.append(output.cpu().numpy())

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
test_metrics_df.to_csv(os.path.join(metrics_dir, 'test_metrics2222.csv'), index=False)

print(f'Test MAE: {test_mae:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test R2: {test_r2:.4f}')
