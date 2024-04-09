import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, random_split
import joblib
import pandas as pd


class CycloneDataset(Dataset):
    def __init__(self, data):
        X = torch.tensor(data[0], dtype=torch.float32)
        Y = torch.tensor(data[1], dtype=torch.float32)

        self.x = X
        self.y = Y
        self.n_samples = Y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def __size__(self):
        return np.shape(self.x), np.shape(self.y)


class Net(nn.Module):
    def __init__(self, channels):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(128 * 10 * 12, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluation(loader, model):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_loss = 0.0
        pred = []
        base = []
        N = 0
        for i, (inputs, labels) in enumerate(loader, 1):
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            pred.append(outputs.numpy())
            base.append(labels.numpy())
            total_loss += loss.item() * len(labels)
            N += len(labels)

        # Print training loss after each epoch
        print(f'Total loss: {total_loss}, N: {N}, Average loss: {total_loss / N}')

    base = np.concatenate(base)
    pred = np.concatenate(pred)
    mse = np.sum((base - pred) ** 2) / len(pred)
    print(f'MSE: {mse}')

    return base, pred


def load_data(list_years, norm, features):
    df = pd.DataFrame()
    if norm == False:
        for y in list_years:
            data = joblib.load(
                'C:/Users/shrei/OneDrive/Documents/Master_Geo/Machine_Learning/data/' + str(y) + '.joblib')
            df = pd.concat([df, data])

    if norm == True:
        for y in list_years:
            data = joblib.load(
                'C:/Users/shrei/OneDrive/Documents/Master_Geo/Machine_Learning/data/' + str(y) + '_normal.joblib')
            df = pd.concat([df, data])

    N = df.shape[0]
    X_data = np.zeros((N, len(features), 24, 28))
    y_data = np.array(df['intensity_next'].values)

    for i in range(N):
        for j, c in enumerate(features):
            X_data[i, j, :, :] = df[c].iloc[i]

    return X_data, y_data


def get_user_norm_input():
    user_input = input("Please type True or False for normalized data: ")
    print('\n')
    if user_input.lower() == 'true':
        return True
    elif user_input.lower() == 'false':
        return False
    else:
        print("Invalid input. Please type either True or False.")
        return get_user_norm_input()


def preparing_data_for_training(Percent_training):
    print("\033[1m" + "User inputs" + "\033[0m")
    year_range = np.arange(int(input("Select starting year: ")), int(input("Select ending year: ")) + 1)
    normalized = get_user_norm_input()
    features = ['intensity', 'va_850', 'tcw', 'skt', 'ua_850']
    x_np, y_np = load_data(year_range, normalized, features)  # Returns input and output as np.array
    dataset = CycloneDataset([x_np, y_np])  # Returns input and output as torch.tensor
    x = dataset.x
    y = dataset.y
    N = dataset.n_samples

    # Combine your input data into a single dataset
    tensor_dataset = TensorDataset(x, y)

    # Determine the lengths of your training and testing sets
    train_len = int(Percent_training * len(tensor_dataset))
    test_len = len(tensor_dataset) - train_len

    # Split the data
    train_data, test_data = random_split(tensor_dataset, [train_len, test_len])

    return train_data, test_data, N, x_np, y_np


def training_model(num_epochs, model, train_loader, optimizer, criterion):
    # Train the model
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        N = 0
        for i, (inputs, labels) in enumerate(train_loader, 1):  # Make sure to unpack float_data from your data loader
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            N += len(labels)

        # Print training loss after each epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Total loss: {total_loss}, N: {N}, Average loss: {total_loss / N}')

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training Duration: {training_duration} seconds")


def choosing_loader_for_evaluation(tr_loader, te_loader, m, x_np, y_np):
    choice = input(
        "Enter 'train' to evaluate train_loader, 'test' to evaluate test_loader, or 'both' to evaluate both: ")
    print_choice = input(
        "Enter (y/n) for yes or no printing: ")
    print('\n')

    if choice == 'train':
        print("Evaluate train loader")
        b, p = evaluation(tr_loader, m)
        print('\n')

        if print_choice == 'y':
            plot_function(x_np, y_np, b, p, choice)


    elif choice == 'test':
        print("Evaluate test loader")
        b, p = evaluation(te_loader, m)
        print('\n')
        if print_choice == 'y':
            plot_function(x_np, y_np, b, p, choice)

    elif choice == 'both':
        print("Evaluate train loader")
        b, p = evaluation(tr_loader, m)
        print('\n')

        if print_choice == 'y':
            plot_function(x_np, y_np, b, p, 'train')
        print('\n')

        print("Evaluate test loader")
        b, p = evaluation(te_loader, m)
        print('\n')

        if print_choice == 'y':
            plot_function(x_np, y_np, b, p, 'test')
    else:
        print("Invalid choice. Please enter 'train', 'test', or 'both'.")


def plot_function(x, y, base, pred, c):
    print("\033[1m" + " Plotting" + "\033[0m")
    print('\n')
    plt.scatter(x[:, 0, 0, 0], y, c='r', label='Persistence')
    plt.plot(x[:, 0, 0, 0], x[:, 0, 0, 0], c='k')
    plt.scatter(base, pred, c='b', label='Model')
    plt.xlabel('Real Outputs (' + c + ')')
    plt.ylabel('Model Predictions (' + c + ')')
    plt.title('Model Predictions vs Real Outputs (' + c + ') and Naive model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Model parameters
    Learning_rate = 0.00001
    Batch_size = 50
    Percent_training_set = 0.8
    Num_epochs = 1

    train_data, test_data, num_samples, x_numpy, y_numpy = preparing_data_for_training(Percent_training_set)

    # Now you can create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=Batch_size, shuffle=False)
    data, _ = next(iter(train_loader))
    num_channels = data.size(1)

    print("\033[1m" + "Information and model parameters" + "\033[0m")
    print("Number of samples:  ", num_samples)
    print("Number of channels:  ", num_channels)
    print("Learning Rate:  ", Learning_rate)
    print("Batch Size:  ", Batch_size)
    print("Percent training set:  ", Percent_training_set)
    print("Number of epochs:  ", Num_epochs)
    print("Result of Naive model: ", np.mean((y_numpy - x_numpy[:, 0, 0, 0].flatten()) ** 2))

    print('\n')

    model = Net(num_channels)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate)

    print("\033[1m" + "Training" + "\033[0m")

    training_model(Num_epochs, model, train_loader, optimizer, criterion)
    print('\n')

    print("\033[1m" + " Evaluation" + "\033[0m")

    choosing_loader_for_evaluation(train_loader, test_loader, model, x_numpy, y_numpy)
