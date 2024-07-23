import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

# Load the sequences from the file
with open('data_sequences120.pkl', 'rb') as f:
    data = pickle.load(f)

train_joint_sequences = data['train_joint_sequences']
val_joint_sequences = data['val_joint_sequences']
test_joint_sequences = data['test_joint_sequences']
train_speech_sequences = data['train_speech_sequences']
val_speech_sequences = data['val_speech_sequences']
test_speech_sequences = data['test_speech_sequences']

# Convert all data to double precision
train_joint_sequences = train_joint_sequences.double()
val_joint_sequences = val_joint_sequences.double()
test_joint_sequences = test_joint_sequences.double()
train_speech_sequences = train_speech_sequences.double()
val_speech_sequences = val_speech_sequences.double()
test_speech_sequences = test_speech_sequences.double()

# Normalize function for X
def normalize(data, bounds):
    data = data.clone()
    if data.dim() == 1:
        data = data.unsqueeze(0)
    dims = data.shape[1]
    for dim in range(dims):
        mn, mx = bounds[0][dim].item(), bounds[1][dim].item()
        data[:, dim] = (data[:, dim] - mn) / (mx - mn)
    return data

# Unnormalize function for X
def unnormalize(data, bounds):
    data = data.clone()
    if data.dim() == 1:
        data = data.unsqueeze(0)
    dims = data.shape[1]
    for dim in range(dims):
        mn, mx = bounds[0][dim].item(), bounds[1][dim].item()
        data[:, dim] = data[:, dim] * (mx - mn) + mn
    return data.squeeze()

# Standardize function for y
def standardize(data, mean, std):
    return (data - mean) / std

# Unstandardize function for y
def unstandardize(data, mean, std):
    return data * std + mean

# Define Encoder, Decoder, and Seq2Seq classes
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout).double()
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout).double()
        self.fc = nn.Linear(hidden_dim, output_dim).double()
        
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
        encoder_outputs, hidden, cell = self.encoder(src)
        outputs, _, _ = self.decoder(trg, hidden, cell)
        return outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the objective function
def objective(config):
    hidden_dim = round(config[0].item())
    num_layers = round(config[1].item())
    num_epochs = round(config[2].item())
    batch_size = round(config[3].item())
    batch_size = 2 ** batch_size
    dropout = config[4].item()
    opt_lr = config[5].item()
    
    print(f"Objective function called with config: hidden_dim={hidden_dim}, num_layers={num_layers}, "
          f"num_epochs={num_epochs}, batch_size={batch_size}, dropout={dropout}, opt_lr={opt_lr}")

    # Create DataLoader with suggested batch_size
    train_loader = DataLoader(TensorDataset(train_speech_sequences, train_joint_sequences), batch_size=batch_size, shuffle=True)
    
    # Initialize models with suggested hyperparameters
    input_dim = train_speech_sequences.shape[2]
    output_dim = train_joint_sequences.shape[2]
    encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(output_dim, hidden_dim, num_layers, dropout)
    model = Seq2Seq(encoder, decoder).to(device).double()

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)

    criterion = nn.MSELoss()
    
    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop with more epochs and early stopping
    for epoch in range(1):
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
        train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
        train_r2 = r2_score(y_true_train, y_pred_train)

        print(f'Epoch {epoch}/{num_epochs}, Training MSE: {epoch_loss:.4f}, '
              f'Training MAE: {train_mae:.4f}, Training RMSE: {train_rmse:.4f}, Training R2: {train_r2:.4f}')
    
        # Evaluate the model on validation data
        with torch.no_grad():
            val_outputs = model(val_speech_sequences.to(device), val_joint_sequences.to(device))
            val_joint_sequences_flat = val_joint_sequences.view(-1, val_joint_sequences.shape[-1]).cpu().numpy()
            val_outputs_flat = val_outputs.view(-1, val_outputs.shape[-1]).cpu().numpy()
            val_loss = mean_squared_error(val_joint_sequences_flat, val_outputs_flat)
            val_rmse = np.sqrt(mean_squared_error(val_joint_sequences_flat, val_outputs_flat))
            val_mae = mean_absolute_error(val_joint_sequences_flat, val_outputs_flat)
            val_r2 = r2_score(val_joint_sequences_flat, val_outputs_flat)
            print(f'Epoch {epoch}/{num_epochs}, Validation MSE: {val_loss:.4f},'
                  f'Validation MAE: {val_mae}, Validation RMSE: {val_rmse}, Validation R2: {val_r2}')
                  
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping")
            break
    
    print(f"Objective function finished with best_val_loss: {best_val_loss}")
    return best_val_loss

# Define the search space
bounds = torch.tensor([
    [64, 2, 10, 1, 0.3, 0.0005],   # Lower bounds
    [256, 3, 25, 6, 0.5, 0.01]  # Upper bounds
], dtype=torch.float64).to(device)

gen_bounds = torch.tensor([
    [0, 0, 0, 0, 0, 0],  # Lower bounds for normalized space
    [1, 1, 1, 1, 1, 1]   # Upper bounds for normalized space
], dtype=torch.float64).to(device)

# Number of initial random points
num_initial_points = 2

# Generate initial random points
X = torch.rand(num_initial_points, bounds.shape[1], dtype=torch.float64).to(device)
initial_hyperparams = X.cpu().numpy().tolist()
print("Initial hyperparameters:")
for i, hyperparams in enumerate(initial_hyperparams):
    print(f"Initial Point {i + 1}: {hyperparams}")

# Evaluate objective function at initial points
y = []
for x in X:
    unnorm_x = unnormalize(x, bounds)
    y.append(objective(unnorm_x))
    print(f"Evaluated objective function for initial point: {unnorm_x}")

y = torch.tensor(y, dtype=torch.float64).to(device)  # Ensure y has shape [n]


# Debugging statement for y
print(f"Initial y values: {y}")
print(f"Shape of y: {y.shape}")

# Fit a GP model
mean, std = y.mean(dim=0), y.std(dim=0)  # Ensure correct dimensions
train_y = standardize(y, mean, std)  # Flatten to match GP input requirements


# Debugging statement for train_y
print(f"Standardized y values: {train_y}")
print(f"Shape of train_y: {train_y.shape}")

gp = SingleTaskGP(X, train_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# Define the acquisition function
acq_func = ExpectedImprovement(gp, best_f=train_y.min(), maximize=False)

# To store hyperparameters and results
results = []
best_result = {'config': None, 'mse': float('inf')}
results_file = 'optimization_results2.json'

# Function to save results incrementally
def save_results(results, best_result, filename):
    data = {
        'results': results,
        'best_result': best_result
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Bayesian Optimization Loop
num_iterations = 10
for iteration in range(num_iterations):
    try:
        # Optimize the acquisition function
        candidates, _ = optimize_acqf(
            acq_func,
            bounds=gen_bounds,
            q=1,
            num_restarts=10,  # Increased for better optimization
            raw_samples=50,   # Increased for better initialization
        )
        
        # Evaluate the objective function at the new candidate
        new_x = candidates.detach()
        new_y = torch.tensor([objective(unnormalize(new_x[0], bounds))], dtype=torch.float64).to(device)



        print(f"Iteration {iteration + 1}/{num_iterations}, Hyperparameters: {new_x.cpu().numpy().tolist()}")
        
        # Update the training data
        X = torch.cat([X, new_x], dim=0)
        y = torch.cat([y, new_y], dim=0) # Flatten y to match GP input requirements
        # y = torch.reshape(y,(y.shape[0],))
        
        true_x = unnormalize(new_x, bounds)
        # Save the results of this iteration
        result = {
            'iteration': iteration + 1,
            'config': true_x.cpu().numpy().tolist(),
            'mse': new_y.item()
        }
        results.append(result)
        
        # Update best result
        if new_y.item() < best_result['mse']:
            best_result['config'] = true_x.cpu().numpy().tolist()
            best_result['mse'] = new_y.item()
        
        # Save results incrementally
        save_results(results, best_result, results_file)
        
        # Debugging statements for the shapes of X and y
        print(f"Shape of X after iteration {iteration + 1}: {X.shape}")
        print(f"Shape of y after iteration {iteration + 1}: {y.shape}")
        
        # Re-fit the GP model
        mean, std = y.mean(dim=0), y.std(dim=0)  # Ensure correct dimensions
        train_y = standardize(y, mean, std) # Flatten to match GP input requirements

        gp.set_train_data(X, train_y, strict=False)
        fit_gpytorch_model(mll)
        
        # Update the acquisition function with the new best observed value
        acq_func = ExpectedImprovement(gp, best_f=train_y.min())
        
        print(f"Iteration {iteration + 1}/{num_iterations}, Best MSE: {y.min().item()}")
    except Exception as e:
        print(f"An error occurred at iteration {iteration + 1}: {e}")
        save_results(results, best_result, results_file)
        break

# Final save
save_results(results, best_result, results_file)
print(f"Optimization results saved to '{results_file}'")