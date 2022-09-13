# -*- coding: utf-8 -*-
# RUN IN PYTHON 3

import os
import cv2
import csv
import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
from pickle import dump
from sklearn import preprocessing
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

dataset_path = "/home/kia/Kiyanoush/UoLincoln/Projects/Tactile_control/data_set/"
# Hyper-parameters:
train_data_dir = dataset_path + 'train/'
test_data_dir  = dataset_path + 'test/'
train_out_dir  = dataset_path + 'train_image_dataset_10c_T10/'
test_out_dir   = dataset_path + 'test_image_dataset_10c_T10/'
scaler_out_dir = dataset_path + 'scalars_10c_T10/'

smooth = False
filter_kalman = True
context_length = 10
horrizon_length = 10


class data_formatter:
    def __init__(self):
        self.files_train = []
        self.files_test = []
        self.full_data_tactile = []
        self.full_data_robot = []
        self.smooth = smooth
        self.filter_kalman = filter_kalman
        self.context_length = context_length
        self.horrizon_length = horrizon_length

    def create_map(self):
        for stage in [train_out_dir, test_out_dir]:
            self.path_file = []
            index_to_save = 0
            print(stage)
            if stage == train_out_dir:
                files_to_run = self.files_train
            else:
                files_to_run = self.files_test

            for experiment_number, file in tqdm(enumerate(files_to_run)):
                if stage == test_out_dir:
                    path_save = stage + "test_trial_" + str(experiment_number) + '/'
                    os.mkdir(path_save)
                    self.path_file = []
                    index_to_save = 0
                else:
                    path_save = stage

                tactile, robot, meta, slip, failure = self.load_file_data(file)

                # scale the data
                for index, (standard_scaler, min_max_scalar) in enumerate(zip(self.tactile_standard_scaler, self.tactile_min_max_scalar)):
                    tactile[:, index] = standard_scaler.transform(tactile[:, index])
                    tactile[:, index] = min_max_scalar.transform(tactile[:, index])
                
                for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
                    robot[:, index] = np.squeeze(min_max_scalar.transform(robot[:, index].reshape(-1, 1)))
                

                sequence_length = self.context_length + self.horrizon_length
                for time_step in range(len(tactile) - sequence_length):
                    robot_data_sequence = [robot[time_step + t] for t in range(sequence_length)]
                    tactile_data_sequence     = [tactile[time_step + t] for t in range(sequence_length)]
                    experiment_data_sequence  = experiment_number
                    time_step_data_sequence   = [time_step + t for t in range(sequence_length)]
                    slip_label_sequence = slip[time_step + sequence_length - 1]
                    failure_label_sequence = failure[time_step + sequence_length - 1]
                  
                    ###################################### Save the data and add to the map ###########################################
                    np.save(path_save + 'robot_data_' + str(index_to_save), robot_data_sequence)
                    np.save(path_save + 'tactile_data_sequence_' + str(index_to_save), tactile_data_sequence)
                    np.save(path_save + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
                    np.save(path_save + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
                    np.save(path_save + 'trial_meta_' + str(index_to_save), np.array(meta))
                    np.save(path_save + 'slip_data_' + str(index_to_save), np.array(slip_label_sequence))
                    np.save(path_save + 'failure_data_' + str(index_to_save), np.array(failure_label_sequence))
                    ref = []
                    ref.append('robot_data_' + str(index_to_save) + '.npy')
                    ref.append('tactile_data_sequence_' + str(index_to_save) + '.npy')
                    ref.append('experiment_number_' + str(index_to_save) + '.npy')
                    ref.append('time_step_data_' + str(index_to_save) + '.npy')
                    ref.append('trial_meta_' + str(index_to_save) + '.npy')
                    ref.append('slip_data_' + str(index_to_save) + '.npy')
                    ref.append('failure_data_' + str(index_to_save) + '.npy')
                    self.path_file.append(ref)
                    index_to_save += 1

                if stage == test_out_dir:
                    self.test_no = experiment_number
                    self.save_map(path_save, test=True)

            self.save_map(path_save)

    def save_map(self, path, test=False):
        if test:
            with open(path + '/map_' + str(self.test_no) + '.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path', 'tactile_data_sequence', 'experiment_number', 'time_steps', 'meta', 'slip', 'failure'])
                for row in self.path_file:
                    writer.writerow(row)
        else:
            with open(path + '/map.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path', 'tactile_data_sequence', 'experiment_number', 'time_steps', 'meta', 'slip', 'failure'])
                for row in self.path_file:
                    writer.writerow(row)

    def scale_data(self):
        files = self.files_train + self.files_test
        for file in tqdm(files):
            tactile, robot, _, _, _ = self.load_file_data(file)
            self.full_data_tactile += list(tactile)
            self.full_data_robot += list(robot)

        self.full_data_tactile = np.array(self.full_data_tactile)
        self.full_data_robot = np.array(self.full_data_robot)

        self.robot_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.full_data_robot[:, feature].reshape(-1, 1)) for feature in range(2)]
        self.tactile_standard_scaler = [preprocessing.StandardScaler().fit(self.full_data_tactile[:, feature]) for feature in range(3)]
        self.tactile_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.tactile_standard_scaler[feature].transform(self.full_data_tactile[:, feature])) for feature in range(3)]

        self.save_scalars()

    def load_file_data(self, file):
        robot_state = np.array(pd.read_csv(file + '/robot_state.csv', header=None))
        meta_data = np.array(pd.read_csv(file + '/meta_data.csv', header=None))
        xela_sensor = pd.read_csv(file + '/xela_sensor1.csv')
        slip_label = pd.read_csv(file + '/slip_label.csv')['slip']
        fail_label = pd.read_csv(file + '/slip_label.csv')['fail']

        xela_sensor = np.array(xela_sensor.sub(xela_sensor.loc[0]))

        robot_task_space = np.array([[state[-3], state[-2]] for state in robot_state[1:]]).astype(float)

        tactile_data_split = [np.array(xela_sensor[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
        tactile_data = np.array([[tactile_data_split[feature][ts] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]) #shape: lenx3x16
       
        if self.filter_kalman:
            for i in range(tactile_data.shape[1]):
                if i < 2: # x, y initial cov = 4
                    kf = KalmanFilter(initial_state_mean=tactile_data[0, i, :], n_dim_obs=16, initial_state_covariance=1*np.eye(16), observation_covariance=4*np.eye(16))
                    tactile_data[:, i, :], _ = kf.filter(tactile_data[:, i, :])
                elif i ==2: # z initial cov = 8
                    kf = KalmanFilter(initial_state_mean=tactile_data[0, i, :], n_dim_obs=16, initial_state_covariance=1*np.eye(16), observation_covariance=8*np.eye(16))
                    tactile_data[:, i, :], _ = kf.filter(tactile_data[:, i, :])
        

        return tactile_data, robot_task_space, meta_data, slip_label, fail_label

    def load_file_names(self):
        self.files_train = glob.glob(train_data_dir + '/*')
        self.files_test = glob.glob(test_data_dir + '/*')

    def save_scalars(self):
        # save the scalars
        dump(self.tactile_standard_scaler[0], open(scaler_out_dir + 'tactile_standard_scaler_x.pkl', 'wb'))
        dump(self.tactile_standard_scaler[1], open(scaler_out_dir + 'tactile_standard_scaler_y.pkl', 'wb'))
        dump(self.tactile_standard_scaler[2], open(scaler_out_dir + 'tactile_standard_scaler_z.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[0], open(scaler_out_dir + 'tactile_min_max_scalar_x.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[1], open(scaler_out_dir + 'tactile_min_max_scalar_y.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[2], open(scaler_out_dir + 'tactile_min_max_scalar_z.pkl', 'wb'))


        dump(self.robot_min_max_scalar[0], open(scaler_out_dir + 'robot_min_max_scalar_vx.pkl', 'wb'))
        dump(self.robot_min_max_scalar[1], open(scaler_out_dir + 'robot_min_max_scalar_vy.pkl', 'wb'))


def main():
    df = data_formatter()
    df.load_file_names()
    df.scale_data()
    df.create_map()


if __name__ == "__main__":
    main()
