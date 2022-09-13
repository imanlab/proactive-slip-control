# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import copy
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use gpu if available
torch.cuda.set_per_process_memory_fraction(fraction=0.8, device=None)
torch.cuda.set_device(device)

# This is the model used for CoRL paper
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
        self.dropout = nn.Dropout(p=0.5).to(device)
        self.norm_layer = nn.LayerNorm(124).to(device)

    def forward(self, tactiles, actions):
        batch_size__ = tactiles.shape[1]
        hidden = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))
        lstm_out, self.hidden_lstm = self.lstm(tactiles, hidden)
        out0 = self.fc0(actions)
        tactile_and_action = torch.cat((lstm_out[-1], out0), 1)
        fcmid_out = self.fcmid(tactile_and_action)
        out1_drop = self.norm_layer(fcmid_out)
        fc1_out = self.relu_activation(self.fc1(out1_drop))
        fc1_out_drop = self.dropout(fc1_out)
        fc2_out = self.tanh_activation(self.fc2(fc1_out_drop))

        return fc2_out


class BatchGenerator:
    def __init__(self, train_percentage, train_data_dir, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_data_dir = train_data_dir
        self.train_percentage = train_percentage
        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, self.train_percentage, self.train_data_dir, self.image_size, train=True)
        dataset_validate = FullDataSet(self.data_map, self.train_percentage, self.train_data_dir, self.image_size, validation=True)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=6, drop_last=True)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=self.batch_size, shuffle=True, num_workers=6, drop_last=True)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train_percentage, train_data_dir, image_size, train=False, validation=False):
        self.train_data_dir = train_data_dir
        self.image_size = image_size
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.train_data_dir + value[0])
        slip_label = np.load(self.train_data_dir + value[5])
        failure_label = np.load(self.train_data_dir + value[6])

        if self.image_size == 0:
            tactile_data = np.load(self.train_data_dir + value[1])
            experiment_number = np.load(self.train_data_dir + value[2])
            time_steps = np.load(self.train_data_dir + value[3])
        else:
            tactile_data = []
            for image_name in np.load(self.train_data_dir + value[1]):
                tactile_data.append(np.load(self.train_data_dir + image_name))
            tactile_data = np.array(tactile_data)
            experiment_number = np.load(self.train_data_dir + value[3])
            time_steps = np.load(self.train_data_dir + value[4])

        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps, slip_label, failure_label]


class UniversalModelTrainer:
    def __init__(self, model, criterion, image_size, model_save_path, model_name, epochs, batch_size,
                 learning_rate, context_frames, sequence_length, train_percentage, validation_percentage, train_data_dir):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.context_frames = context_frames
        self.sequence_length = sequence_length
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.model = model
        self.image_size = image_size
        self. train_data_dir = train_data_dir

        BG = BatchGenerator(self.train_percentage, self.train_data_dir, self.batch_size, self.image_size)
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        if criterion == "L1":
            self.criterion = nn.L1Loss()
        if criterion == "L2":
            self.criterion = nn.MSELoss()
        if criterion == "BCEWithLogitsLoss":
            weight = torch.Tensor([4]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_full_model(self):
        best_training_loss = 100.0
        training_val_losses = []
        training_val_acc = []
        progress_bar = tqdm(range(0, self.epochs))
        for epoch in progress_bar:
            model_save = ""
            self.train_loss = 0.0
            self.val_loss = 0.0
            self.train_acc = 0.0
            self.val_acc = 0.0

            # trainging
            for index, batch_features in enumerate(self.train_full_loader):
                self.optimizer.zero_grad()
                # action = batch_features[0].permute(1, 0, 2).to(device)
                action = torch.flatten(batch_features[0][:, :10, :], start_dim=1).to(device)
                slip_label = batch_features[4].to(device)
                failure_label = batch_features[5].to(device)
                tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2)[:10, :, :32].to(device)
                loss = self.run_batch(action, tactile, slip_label, train=True)            
                train_max_index = index

            # validation
            for index, batch_features in enumerate(self.valid_full_loader):
                self.optimizer.zero_grad()
                # action = batch_features[0].permute(1, 0, 2).to(device)
                action = torch.flatten(batch_features[0][:, :10, :], start_dim=1).to(device)
                slip_label = batch_features[4].to(device)
                failure_label = batch_features[5].to(device)
                tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2)[:10, :, :32].to(device)
                loss = self.run_batch(action, tactile, slip_label, validation=True)    
                val_max_index = index

            training_val_losses.append([self.train_loss/(train_max_index+1), self.val_loss/(val_max_index+1)])
            training_val_acc.append([self.train_acc/(train_max_index+1), self.val_acc/(val_max_index+1)])
            np.save(self.model_save_path + "train_val_losses", np.array(training_val_losses))
            np.save(self.model_save_path + "train_val_accuracy", np.array(training_val_acc))

            # early stopping and saving:
            if best_training_loss > self.val_loss/(val_max_index+1):
                best_training_loss = self.val_loss/(val_max_index+1)
                torch.save(self.model, self.model_save_path + self.model_name)
                model_save = "saved model"

            print("Training mean loss: {:.4f} || Validation mean loss: {:.4f} || Training mean Acc: {:.4f} || Validation mean Acc: {:.4f} || {}".format(self.train_loss/(train_max_index+1), self.val_loss/(val_max_index+1), self.train_acc/(train_max_index+1), self.val_acc/(val_max_index+1), model_save))

    def run_batch(self, action, tactile, slip_label, train=False, validation=False):
        # print(self.model.device)
        slip_predictions = self.model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.
        loss = self.criterion(slip_predictions, slip_label.unsqueeze(1))
        acc = self.binary_acc(slip_predictions, slip_label.unsqueeze(1))

        if train:
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item()
            self.train_acc += acc.item()
        elif validation:
            self.val_loss += loss.item()
            self.val_acc += acc.item()

        return loss.item()
    
    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
        
        return acc

def main():
    model_save_path = "/home/kia/Kiyanoush/UoLincoln/Projects/Tactile_control/Humanoids/saved_models/test_slip_prediction/"
    train_data_dir = "/home/kia/Kiyanoush/UoLincoln/Projects/Tactile_control/data_set/train_image_dataset_10c_10h/"

    # unique save title:
    model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
    os.mkdir(model_save_path)

    epochs = 60
    batch_size = 32
    learning_rate = 1e-3
    context_frames = 10
    sequence_length = 20
    train_percentage = 0.9
    validation_percentage = 0.1
    image_size = 0  # set to zero if linear data
    criterion = "BCEWithLogitsLoss"
    model_name = "Humanoids_AC_LSTM"
    model = ClassifierLSTM(device=device, context_frames=context_frames)
    model.to(device)

    UMT = UniversalModelTrainer(model, criterion, image_size, model_save_path, model_name,
                          epochs, batch_size, learning_rate, context_frames, sequence_length,
                          train_percentage, validation_percentage, train_data_dir)
    UMT.train_full_model()

if __name__ == "__main__":
    main()
