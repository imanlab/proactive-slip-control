# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

class ClassifierLSTM(nn.Module):
    
    def __init__(self, device, context_frames):
        super(ClassifierLSTM, self).__init__()
        self.device = device
        self.context_frames = context_frames
        self.lstm = nn.LSTM(32, 200).to(device)  # tactile
        self.fc0 = nn.Linear(20, 48).to(device)
        self.fcmid = nn.Linear(248, 124).to(device)
        self.fc1 = nn.Linear(124, 40).to(device)
        self.fc2 = nn.Linear(40, 1).to(device)
        self.tanh_activation = nn.Tanh().to(device)
        self.relu_activation = nn.ReLU().to(device)
        self.softmax_activation = nn.Softmax(dim=1).to(device) #we don't use this because BCE loss in pytorch automatically applies the the Sigmoid activation
        self.dropout = nn.Dropout(p=0.5)
        self.norm_layer = nn.LayerNorm(124)

    def forward(self, tactiles, actions):
        batch_size__ = tactiles.shape[1]
        hidden = (torch.zeros(1, batch_size__, 200, device=torch.device('cpu')), torch.zeros(1, batch_size__, 200, device=torch.device('cpu')))
        lstm_out, self.hidden_lstm = self.lstm(tactiles, hidden)
        out0 = self.fc0(actions)
        tactile_and_action = torch.cat((lstm_out[-1], out0), 1)
        fcmid_out = self.fcmid(tactile_and_action)
        out1_drop = self.norm_layer(fcmid_out)
        fc1_out = self.relu_activation(self.fc1(out1_drop))
        fc1_out_drop = self.dropout(fc1_out)
        fc2_out = self.tanh_activation(self.fc2(fc1_out_drop))

        return fc2_out