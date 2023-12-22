import torch
import torch.nn.functional as F  
from torch import nn
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.init as init
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

class NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_sizes, activation_function):
        super(NN, self).__init__()

        seed = 18
        torch.manual_seed(seed)

        self.activation = activation_function

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_layer_sizes[0])
        init.kaiming_uniform_(self.input_layer.weight, mode='fan_in', nonlinearity=activation_function.__name__)


        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            for i in range(len(hidden_layer_sizes) - 1)
        ])

        for layer in self.hidden_layers:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity=activation_function.__name__)


        # Output layer
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], num_classes)
        init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity=activation_function.__name__)


    def forward(self, x):
        x = x.float()
        x = self.activation(self.input_layer(x))

        # Process through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x = self.output_layer(x)

        return x
    

def train_model(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total loss for the epoch
        num_batches = len(train_loader)

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.reshape(data.shape[0], -1)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate the batch loss

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    all_predictions = []
    all_targets = []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)
            scores = model(x)

            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    model.train()
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    accuracy = num_correct / num_samples
    return accuracy, f1


def train_test_split(data: pd.DataFrame, target_label : str, test_size=0.2, return_torch=True, DoSmote = False):
        
    # split the data into train and test
    train = data.sample(frac=(1-test_size),random_state=200)
    test = data.drop(train.index)
    
    # split the train and test into X and Y
    train_X = train.drop([target_label], axis=1).values
    train_Y = train[target_label].values
    if DoSmote == True:
        sm = SMOTE(random_state=42, k_neighbors= 3)
        train_X, train_Y = sm.fit_resample(train_X, train_Y)
    test_X = test.drop([target_label], axis=1).values
    test_Y = test[target_label].values
    
    if return_torch:
        train_X = torch.tensor(train_X)
        train_Y = torch.tensor(train_Y)
        test_X = torch.tensor(test_X)
        test_Y = torch.tensor(test_Y)
    
    return train_X, train_Y, test_X, test_Y