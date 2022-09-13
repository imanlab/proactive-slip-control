import torch
import torch.nn as nn

class ClassifierLSTM(nn.Module):
    def __init__(self, device, context_frames):
        super(ClassifierLSTM, self).__init__()
        self.device = device
        self.context_frames = context_frames
        self.lstm = nn.LSTM(32, 200).to(device)  # tactile
        self.fc1 = nn.Linear(200, 40).to(device)
        self.fc2 = nn.Linear(40, 1).to(device)
        self.tan_activation = nn.Tanh().to(device)
        self.relu_activation = nn.ReLU().to(device)
        self.softmax_activation = nn.Softmax(dim=1).to(device) #we don't use this because BCE loss in pytorch automatically applies the the Sigmoid activation
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, tactiles):
        batch_size__ = tactiles.shape[1]
        hidden = (torch.zeros(1, batch_size__, 200, device=torch.device('cpu')), torch.zeros(1, batch_size__, 200, device=torch.device('cpu')))
        lstm_out, self.hidden_lstm = self.lstm(tactiles, hidden)
        lstm_out_drop = self.dropout(lstm_out[-1])
        fc1_out = self.relu_activation(self.fc1(lstm_out_drop))
        fc1_out_drop = self.dropout(fc1_out)
        fc2_out = self.tan_activation(self.fc2(fc1_out_drop))

        return fc2_out