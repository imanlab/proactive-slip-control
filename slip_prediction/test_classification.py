import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
from pickle import dump, load
from sklearn import preprocessing
from scipy.spatial.transform import Rotation as R
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


context_length = 10
horrizon_length = 10

scaler_path = '/home/kia/Kiyanoush/UoLincoln/Projects/Tactile_control/data_set/scalars'
robot_min_max_scalar    = [load(open(scaler_path + '/robot_min_max_scalar_'+feature +'.pkl', 'rb')) for feature in ['vx', 'vy']]
tactile_standard_scaler = [load(open(scaler_path + '/tactile_standard_scaler_'+feature +'.pkl', 'rb')) for feature in ['x', 'y', 'z']]
tactile_min_max_scalar  = [load(open(scaler_path + '/tactile_min_max_scalar_'+feature +'.pkl', 'rb')) for feature in ['x', 'y', 'z']]


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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


model = torch.load("/home/kia/Kiyanoush/UoLincoln/Projects/Tactile_control/Humanoids/saved_models/test_slip_prediction/model_30_06_2022_12_07/Humanoids_AC_LSTM").to(device)

def load_file_data(file):
    robot_state = np.array(pd.read_csv(file + '/robot_state.csv', header=None))
    meta_data = np.array(pd.read_csv(file + '/meta_data.csv', header=None))
    xela_sensor = pd.read_csv(file + '/xela_sensor1.csv')
    slip_label = pd.read_csv(file + '/slip_label.csv')['slip']
    fail_label = pd.read_csv(file + '/slip_label.csv')['fail']

    xela_sensor = np.array(xela_sensor.sub(xela_sensor.loc[0]))

    robot_task_space = np.array([[state[-3], state[-2]] for state in robot_state[1:]]).astype(float)

    tactile_data_split = [np.array(xela_sensor[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
    tactile_data = np.array([[tactile_data_split[feature][ts] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]) #shape: lenx3x16
    
    for i in range(tactile_data.shape[1]):
        if i < 2: # x, y initial cov = 4
            kf = KalmanFilter(initial_state_mean=tactile_data[0, i, :], n_dim_obs=16, initial_state_covariance=1*np.eye(16), observation_covariance=4*np.eye(16))
            tactile_data[:, i, :], _ = kf.filter(tactile_data[:, i, :])
        elif i ==2: # z initial cov = 8
            kf = KalmanFilter(initial_state_mean=tactile_data[0, i, :], n_dim_obs=16, initial_state_covariance=1*np.eye(16), observation_covariance=8*np.eye(16))
            tactile_data[:, i, :], _ = kf.filter(tactile_data[:, i, :])
    

    return tactile_data, robot_task_space, meta_data, slip_label, fail_label


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

files_to_run = sorted(glob.glob("/home/kia/Kiyanoush/UoLincoln/Projects/Tactile_control/data_set/Humanoids_classification/*"))

tactile_full = []
action_full = []
slip_full = []

for experiment_number, file in tqdm(enumerate(files_to_run)):

    tactile, robot, meta, slip, failure = load_file_data(file)

    # scale the data
    for index, (standard_scaler, min_max_scalar) in enumerate(zip(tactile_standard_scaler, tactile_min_max_scalar)):
        tactile[:, index] = standard_scaler.transform(tactile[:, index])
        tactile[:, index] = min_max_scalar.transform(tactile[:, index])
    
    for index, min_max_scalar in enumerate(robot_min_max_scalar):
        robot[:, index] = np.squeeze(min_max_scalar.transform(robot[:, index].reshape(-1, 1)))

    sequence_length = context_length + horrizon_length
    for time_step in range(len(tactile) - sequence_length):
        robot_data_sequence = [robot[time_step + t] for t in range(context_length, sequence_length)]
        tactile_data_sequence     = [tactile[time_step + t, :32] for t in range(context_length)]
       
        slip_label_sequence = slip[time_step + sequence_length - 1]

        tactile_full.append(tactile_data_sequence)
        action_full.append(np.array(robot_data_sequence).flatten())
        slip_full.append(slip_label_sequence)

tactile_full = torch.from_numpy(np.array(tactile_full)).flatten(start_dim=2)[:, :, :32].view(10, -1, 32).to(device).type(torch.float32)
action_full = torch.from_numpy(np.array(action_full)).view(-1, 20).to(device).type(torch.float32)
slip_full = torch.from_numpy(np.array(slip_full)).view(-1, 1).to(device).type(torch.float32)

slip_predictions = model(tactile_full, action_full)

acc = binary_acc(slip_predictions, slip_full)

slip_predictions = torch.round(torch.sigmoid(slip_predictions))
slip_full = slip_full.cpu().detach().numpy()
slip_predictions = slip_predictions.cpu().detach().numpy()

print(classification_report(slip_full, slip_predictions))

conf_matrix = confusion_matrix(slip_full, slip_predictions)
LABELS = ["non_slip","slip"]
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
# plt.show()
plt.savefig("confusionT10.png")
            