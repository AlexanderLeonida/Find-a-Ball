import torch
import torch.nn as nn
import ast
import numpy as np
import matplotlib.pyplot as plt

# multi-step LSTM-based motion prediction model
class MotionPredictorMultiStep(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, future_steps=5):
        super(MotionPredictorMultiStep, self).__init__()
        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # fully connected layer to output all future steps at once
        self.fc = nn.Linear(hidden_size, output_size * future_steps)
        self.future_steps = future_steps
        self.output_size = output_size

    def forward(self, x):
        out, _ = self.lstm(x) # LSTM output over sequence
        out = out[:, -1, :]   # Take the last time step output
        out = self.fc(out)    # Predict future steps from last hidden state
        out = out.view(-1, self.future_steps, self.output_size)  # Reshape to (batch, future_steps, output_size)
        return out

# load training data from a text file and format it for training
def generate_multistep_data(filePath):
    # filePath = input('Input file to train model on: ')
    # print(f"training_data/{filePath}")
    try:
        # parse file content to extract x and y coordinates
        with open(f"training_data/{filePath}", 'r') as file:
            file_list = str(file.read()).split(":")
            # file list stored as string tuples. Ex. "(350, 450)". 
            # x and y coordinates stored as integers
            x_coordinates = [ast.literal_eval(file_list[2*i])[0] for i in range(len(file_list) // 2)]
            y_coordinates = [ast.literal_eval(file_list[2*i + 1])[1] for i in range(len(file_list) // 2 - 1)]
    except FileNotFoundError:
        print("File (input) not found")
        exit()

    # split into training 75% and prediction 25% 
    x_X = [x_coordinates[:int(len(x_coordinates) * 0.75)]]
    x_y = [x_coordinates[int(len(x_coordinates) * 0.75):]]
    y_X = [y_coordinates[:int(len(y_coordinates) * 0.75)]]
    y_y = [y_coordinates[int(len(y_coordinates) * 0.75):]]

    # Convert to PyTorch tensors and add feature dimension
    x_X = torch.tensor(x_X, dtype=torch.float32).unsqueeze(-1)
    x_y = torch.tensor(x_y, dtype=torch.float32).unsqueeze(-1)
    y_X = torch.tensor(y_X, dtype=torch.float32).unsqueeze(-1)
    y_y = torch.tensor(y_y, dtype=torch.float32).unsqueeze(-1)

    return x_X, x_y, y_X, y_y

def train_lstm_model(filePath, num_epochs=1000, lr=0.1):
    # load training and target data
    x_X, x_y, y_X, y_y = generate_multistep_data(filePath)

    # initialize models for predicting x and y motion
    model_x = MotionPredictorMultiStep(input_size=1, output_size=1, future_steps=x_y.shape[1])
    model_y = MotionPredictorMultiStep(input_size=1, output_size=1, future_steps=y_y.shape[1])

    # mean squared error loss function
    criterion = nn.MSELoss()

    # optimizers for both x and y models with learning rate
    # mess with lr parameter for better predictions. a higher lr helps fix loss faster over each epoch faster. 
    # if lr is larger, there's a higher chance the model will never converge to a correct output
    # higher lr basically means higher degree of changing of nodes in back propogation
    optimizer_x = torch.optim.Adam(model_x.parameters(), lr=lr)
    optimizer_y = torch.optim.Adam(model_y.parameters(), lr=lr)

    # train for 300 epochs (good for back_forth_line.txt)
    # should maybe ask user how much they want to train the model. if lr is too high, at a certain point, more epochs becomes redundant
    for epoch in range(num_epochs):
        # train model_x on x-coordinate data
        model_x.train()
        output_x = model_x(x_X)
        loss_x = criterion(output_x, x_y)

        optimizer_x.zero_grad()
        loss_x.backward()
        optimizer_x.step()

        # train model_y on y-coordinate data
        model_y.train()
        output_y = model_y(y_X)
        loss_y = criterion(output_y, y_y)

        optimizer_y.zero_grad()
        loss_y.backward()
        optimizer_y.step()

        # print losses every 10 epochs
        # if losses are decreasing, model is improving. loss = actual - predicted squared
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss X: {loss_x.item():.4f}, Loss Y: {loss_y.item():.4f}")

    # torch.save(model_x.state_dict(), 'models/model_x_weights.pth')
    # torch.save(model_y.state_dict(), 'models/model_y_weights.pth')
    torch.save(model_x, 'models/model_x_full.pth')
    torch.save(model_y, 'models/model_y_full.pth')


    # Set models to evaluation mode
    model_x.eval()
    model_y.eval()

    # predictions with no gradient computation
    with torch.no_grad():
        predicted_x = model_x(x_X).squeeze(-1).numpy()  # predicted x positions
        predicted_y = model_y(y_X).squeeze(-1).numpy()  # predicted y positions
        actual_x = x_y.squeeze(-1).numpy()              # ground truth x positions
        actual_y = y_y.squeeze(-1).numpy()              # ground truth y positions

    # predicted vs actual positions
    print("\nPredicted future x positions:", predicted_x[0])
    print("Actual future x positions   :", actual_x[0])
    print("\nPredicted future y positions:", predicted_y[0])
    print("Actual future y positions   :", actual_y[0])

    # plot the predictions vs actual results
    plt.figure(figsize=(10, 4))

    # X position plot
    plt.subplot(1, 2, 1)
    plt.plot(actual_x[0], label='Actual X')
    plt.plot(predicted_x[0], label='Predicted X', linestyle='--')
    plt.title('Future X Position Prediction')
    plt.legend()

    # Y position plot
    plt.subplot(1, 2, 2)
    plt.plot(actual_y[0], label='Actual Y')
    plt.plot(predicted_y[0], label='Predicted Y', linestyle='--')
    plt.title('Future Y Position Prediction')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # return predicted x and y coordinates
    return predicted_x[0], predicted_y[0]