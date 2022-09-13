#! /usr/bin/env python3
from os import device_encoding
import rospy
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import message_filters
from pickle import load
import numpy.matlib as mat
from numpy.random import seed
from pykalman import KalmanFilter
# from xela_server.msg import XStream
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Float64
from scipy.spatial.transform import Rotation as R

## Optimizer
from scipy.optimize import Bounds
from scipy.spatial import distance
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

## Pytorch
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import onnx
import onnxruntime

from CoRL_AC_SP.AC_SP_model import ClassifierLSTM

from scipy.special import expit


seed = 42
context_frames = 10
sequence_length = 20
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobotTest():
	def __init__(self):
		self.time_step = 0
		self.prev_time_step = 0
		self.stop = 0.0
		self.translation_x = 0.35
		self.translation_y = 0.46
		self.time_step_after_slip = 0
		self.xela_data 		      = np.zeros((700, 48))
		self.xela_kalman_mean	  = np.zeros((700, 48))
		self.xela_kalman_cov_x    = np.zeros((700, 16, 16))
		self.xela_kalman_cov_y    = np.zeros((700, 16, 16))
		self.xela_kalman_cov_z    = np.zeros((700, 16, 16))
		self.xela_scaled_data     = np.zeros((700, 32))
		self.robot_data 	      = np.zeros((700, 2))
		self.action_data 	      = np.zeros((700, 2))
		self.action_data_scaled   = np.zeros((700, 2))
		self.predicted_slip_class = np.zeros((700, 10))
		self.marker_data		  = np.zeros((700, 7))
		
		self.optimal_weights      = np.zeros((700, 4))
		self.optimal_trajectory   = np.zeros((700, 10, 2))
		self.opt_execution_time   = np.zeros(700)
		self.optimality           = np.zeros(700)
		self.num_itr              = np.zeros(700)
		self.constr_violation     = np.zeros(700)
		self.save_results_path    = "/home/kiyanoush/Desktop/CoRL_videos/test_obj/Monster/PSC"
		self.previous_w = np.zeros(4)
		
		self.robot_actions_pre_calc = np.load("/home/kiyanoush/Cpp_ws/src/robotTest2/data/RT_test/Test_013/robot_data.npy")
		self.opt_call_fm_count = 0
		self.class_ort_session = onnxruntime.InferenceSession("/home/kiyanoush/Cpp_ws/src/robotTest2/ONNX_torch/CoRL_AC_SP.onnx")

		rospy.init_node('listener', anonymous=True, disable_signals=True)
		self.model_predict_001_init()
		self.load_scalers()
		self.init_sub()
		self.control_loop()
	
	def init_sub(self):
		sync_data_sub = message_filters.Subscriber('/sync_data', Float64MultiArray)
		self.sync_subscriber = [sync_data_sub]
		ts_sync = message_filters.ApproximateTimeSynchronizer(self.sync_subscriber, queue_size=1, slop=0.1, allow_headerless=True)
		ts_sync.registerCallback(self.sub_cb)
		self.slip_prediction_pub = rospy.Publisher('/slip_prediction', Float64, queue_size=11)
		self.optimal_traj_pub = rospy.Publisher('/optimal_traj', Float64MultiArray, queue_size=11)

	def load_scalers(self):
		self.scaler_path = '/home/kiyanoush/Cpp_ws/src/robotTest2/python scripts/CoRL/scalars/AC_model'
		self.robot_min_max_scalar    = [load(open(self.scaler_path + '/robot_min_max_scalar_'+feature +'.pkl', 'rb')) for feature in ['vx', 'vy']]
		self.tactile_standard_scaler = [load(open(self.scaler_path + '/tactile_standard_scaler_'+feature +'.pkl', 'rb')) for feature in ['x', 'y', 'z']]
		self.tactile_min_max_scalar  = [load(open(self.scaler_path + '/tactile_min_max_scalar_'+feature +'.pkl', 'rb')) for feature in ['x', 'y', 'z']]
	
	def descale_data(self, tactile):
		tp_back_scaled = []
		if tactile.shape[0] == 48:
			(tpx, tpy, tpz) = np.split(np.expand_dims(tactile, axis=0), 3, axis=1)
		else:
			(tpx, tpy, tpz) = np.split(tactile, 3, axis=1)

		xela_x_inverse_minmax = self.tactile_min_max_scalar[0].inverse_transform(tpx)
		xela_y_inverse_minmax = self.tactile_min_max_scalar[1].inverse_transform(tpy)
		xela_z_inverse_minmax = self.tactile_min_max_scalar[2].inverse_transform(tpz)
		xela_x_inverse_full = self.tactile_standard_scaler[0].inverse_transform(xela_x_inverse_minmax)
		xela_y_inverse_full = self.tactile_standard_scaler[1].inverse_transform(xela_y_inverse_minmax)
		xela_z_inverse_full = self.tactile_standard_scaler[2].inverse_transform(xela_z_inverse_minmax)
		tp_back_scaled = np.concatenate((xela_x_inverse_full, xela_y_inverse_full, xela_z_inverse_full), axis=1)

		return tp_back_scaled

	def cons_f(self, w):
		trajectory_x = np.matmul(self.Phi_mat, w)
		trajectory_y = trajectory_x * (self.translation_y/self.translation_x)
		self.trajectory = np.concatenate((trajectory_x[:, np.newaxis], trajectory_y[:, np.newaxis]), axis=1)
		self.trajectory = self.scale_action(self.trajectory)

		pred_ort_inputs = {self.class_ort_session.get_inputs()[0].name: self.scaled_tactile, self.class_ort_session.get_inputs()[1].name: self.trajectory}
		prediction = self.class_ort_session.run(None, pred_ort_inputs)
		slip_class = expit(self.prediction[0][0][0])
		slip_pred_tag = np.round(slip_class)

		return slip_pred_tag

	def BasisFuncGauss(self, N, h, dt):
		self.T = int(round(1/dt+1))
		self.Phi = np.zeros((self.T-1,N))
		for z in range(0,self.T-1):
			t = z*dt
			self.phi = np.zeros((1, N))
			for k in range(1,N+1):
				c = (k-1)/(N-1)
				self.phi[0,k-1] = np.exp(-(t - c)*(t - c)/(2*h))
			self.Phi[z,:N] = self.phi[0, :N]
		self.Phi = self.Phi/np.transpose(mat.repmat(np.sum(self.Phi,axis=1),N,1)); #[TxN]
		return self.Phi #[TxN]

	def dist_to_nominal(self, w, Phi_mat, ref):
		""" Euclidean distance to the reference trajectory """
		return np.sqrt(np.sum((np.matmul(Phi_mat, w)-ref)**2))

	def distance_der(self, w, Phi_mat, ref):
		""" Derivative of Euclidean distance w.r.t weight vector """
		return np.matmul(Phi_mat.T, (np.matmul(Phi_mat, w)-ref)) / np.sqrt(np.sum((np.matmul(Phi_mat, w)-ref)**2))

	def gen_opt_traj(self, nom_traj):
		self.nom_traj = nom_traj
		self.num_basis = 4
		self.Phi_mat = self.BasisFuncGauss(self.num_basis, 0.015, dt=0.1)
		w0 = np.zeros(4)
		# w0 = np.random.normal(0, 1, 4)
		# w0 = self.previous_w

		linear_constraint = LinearConstraint([self.Phi_mat[0, :]], self.robot_data[self.time_step-1, 0]-0.15, self.robot_data[self.time_step-1, 0]+0.15)
		nonlinear_constraint = NonlinearConstraint(self.cons_f, 0.0, 0.0)
		optimizer_options = {'verbose': 0, 'xtol':1e-08, 'gtol':1e-08,  'maxiter':9, 'factorization_method': 'SVDFactorization', 'disp':False}
		res = minimize(self.dist_to_nominal, w0, args=(self.Phi_mat, self.nom_traj), method='trust-constr', jac=self.distance_der,
					constraints=[linear_constraint, nonlinear_constraint], options=optimizer_options)

		self.opt_execution_time[self.time_step] = res.execution_time
		self.optimality[self.time_step] = res.optimality
		self.num_itr[self.time_step] = res.nit
		self.num_itr[self.time_step] = res.nit
		self.constr_violation[self.time_step] = res.constr_violation

		self.previous_w = res.x
		
		return res.x
	
	def optimization_results(result):
		pass	

	def remove_offset(self, tactile_data_init):
		tactile_data_init = tactile_data_init - self.initial_xela_row
		tactile_data_split = [np.array(tactile_data_init[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]	
		tactile = [[tactile_data_split[feature][ts] for feature in range(3)] for ts 
																		in range(tactile_data_split[0].shape[0])] # output of shape trial_lenx3x16
		tactile = np.array(tactile)

		return tactile[0] # shape: 3x16
	
	def scale_action(self, action):
		new_action = np.copy(action)
		for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
			new_action[:, index] = np.squeeze(min_max_scalar.transform(new_action[:, index].reshape(-1, 1)))

		# action = torch.from_numpy(action).view(20, 1, 2).double()
		new_action = new_action.flatten()[np.newaxis, :].astype(np.float32)

		return new_action

	def scale_data(self, tactile_data, action):
		new_vec = np.zeros((10, 48))
		new_vec[:, :16] = self.tactile_standard_scaler[0].transform(tactile_data[:, :16])
		new_vec[:, 16:32] = self.tactile_standard_scaler[1].transform(tactile_data[:, 16:32])
		new_vec[:, 32:48] = self.tactile_standard_scaler[2].transform(tactile_data[:, 32:48])
		
		new_vec[:, :16] = self.tactile_min_max_scalar[0].transform(new_vec[:, :16])
		new_vec[:, 16:32] = self.tactile_min_max_scalar[1].transform(new_vec[:, 16:32])
		new_vec[:, 32:48] = self.tactile_min_max_scalar[2].transform(new_vec[:, 32:48])
		
		new_action = np.copy(action)
		for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
			new_action[:, index] = np.squeeze(min_max_scalar.transform(new_action[:, index].reshape(-1, 1)))
		
		tactile_input = new_vec[:, np.newaxis, :32].astype(np.float32)
		# # action = action[:, np.newaxis, :].astype(np.float32)
		new_action = new_action.flatten()[np.newaxis, :].astype(np.float32)

		# tactile_input = torch.from_numpy(tactile_input).view(10, 1, 32)
		# new_action = torch.from_numpy(new_action).view(1, 20)

		return tactile_input, new_action
	
	def preprocess_predictions(self, tactile_prediction):
		tactile_prediction[:, :32] = self.tactile_pred_standard_scaler.transform(tactile_prediction[:, :32])
		
		# tactile_prediction = torch.from_numpy(tactile_prediction[:, :32]).view(10, 1, 32).type(torch.FloatTensor)
		tactile_prediction = tactile_prediction[:, np.newaxis, :32].astype(np.float32)

		return tactile_prediction
	

	def model_predict_001_init(self):
		# Classifier LSTM
		self.classifier = torch.load("/home/kiyanoush/Cpp_ws/src/robotTest2/python scripts/CoRL/torch_models/CoRL_AC_LSTM", map_location='cpu').to(device).float()
		self.classifier.eval()
	

	def init_kalman_filter(self, xela_vec):
		self.kf_x = KalmanFilter(initial_state_mean=xela_vec[0], n_dim_obs=16, transition_covariance=1*np.eye(16), observation_covariance=4*np.eye(16))
		self.kf_y = KalmanFilter(initial_state_mean=xela_vec[1], n_dim_obs=16, transition_covariance=1*np.eye(16), observation_covariance=4*np.eye(16))
		self.kf_z = KalmanFilter(initial_state_mean=xela_vec[2], n_dim_obs=16, transition_covariance=1*np.eye(16), observation_covariance=8*np.eye(16))

		self.xela_kalman_mean[0, 0:16], self.xela_kalman_cov_x[0] = self.kf_x.filter_update(xela_vec[0], 4*np.eye(16), xela_vec[0])
		self.xela_kalman_mean[0, 16:32], self.xela_kalman_cov_y[0] = self.kf_y.filter_update(xela_vec[1], 4*np.eye(16), xela_vec[1])
		self.xela_kalman_mean[0, 32:48], self.xela_kalman_cov_z[0] = self.kf_z.filter_update(xela_vec[2], 8*np.eye(16), xela_vec[2])

	def kalman_filter_func(self):
		if self.time_step == 0:
			self.init_kalman_filter(self.xela_vec)
		else:
			self.xela_kalman_mean[self.time_step, :16], self.xela_kalman_cov_x[self.time_step] = \
					self.kf_x.filter_update(self.xela_kalman_mean[self.time_step-1, :16], self.xela_kalman_cov_x[self.time_step-1], self.xela_vec[0])
			self.xela_kalman_mean[self.time_step, 16:32], self.xela_kalman_cov_y[self.time_step] = \
					self.kf_y.filter_update(self.xela_kalman_mean[self.time_step-1, 16:32], self.xela_kalman_cov_y[self.time_step-1], self.xela_vec[1])
			self.xela_kalman_mean[self.time_step, 32:48], self.xela_kalman_cov_z[self.time_step] = \
					self.kf_z.filter_update(self.xela_kalman_mean[self.time_step-1, 32:48], self.xela_kalman_cov_z[self.time_step-1], self.xela_vec[2])
			
	def add_to_save(self):
		# store the prediction data:
		self.xela_scaled_data[self.time_step, :]   = self.scaled_tactile[-1, 0]
		self.action_data_scaled[self.time_step, :] = self.scaled_action[0 , -2:]
	
	def save_results(self):
		np.save(self.save_results_path + "/xela_raw.npy", self.xela_data[:self.time_step])
		np.save(self.save_results_path + "/xela_kalman_filtered.npy", self.xela_kalman_mean[:self.time_step])
		np.save(self.save_results_path + "/xela_scaled.npy", self.xela_scaled_data[:self.time_step])
		np.save(self.save_results_path + "/slip_prediction.npy", self.predicted_slip_class[:self.time_step])
		np.save(self.save_results_path + "/robot_data.npy", self.robot_data[:self.time_step])
		np.save(self.save_results_path + "/action_data.npy", self.action_data[:self.time_step])
		np.save(self.save_results_path + "/action_data_scaled.npy", self.action_data_scaled[:self.time_step])
		np.save(self.save_results_path + "/marker.npy", self.marker_data[:self.time_step])
		np.save(self.save_results_path + "/trajectory_weights.npy", self.optimal_weights[:self.time_step])
		np.save(self.save_results_path + "/trajectories.npy", self.optimal_trajectory[:self.time_step])
		np.save(self.save_results_path + "/basis_function.npy", self.Phi_mat)
		self.save_meta_data()
	
	def save_meta_data(self):
		data_keys = ['Control_version', 'execution_time', 'Optimality', 'num_iteration', 'constraint_violation', 'Cpp_scale']
		meta_data_dict = dict.fromkeys(data_keys)

		meta_data_dict['Control_version'] = 'AC_SP'
		meta_data_dict['execution_time'] = self.opt_execution_time
		meta_data_dict['Optimality'] = self.optimality
		meta_data_dict['num_iteration'] = self.num_itr
		meta_data_dict['constraint_violation'] = self.constr_violation
		meta_data_dict['Cpp_scale'] = 120
		
		meta_file = open(self.save_results_path + "/data.pkl", "wb")
		pickle.dump(meta_data_dict, meta_file)
		meta_file.close()

	def sub_cb(self, sync_data):

		self.stop = sync_data.data[67]

		if self.time_step == 0:
			self.initial_xela_row = np.array(sync_data.data[:48])[np.newaxis, :]

		if self.stop == 0.0:
			self.slip_onset_vx = sync_data.data[-2]
			self.post_slip_lin_acc_T = sync_data.data[-1]
			self.state_vec = [sync_data.data[64], sync_data.data[65]]
			self.marker_data[self.time_step, :] = sync_data.data[-9:-2]
			try:
				self.current_actions = self.robot_actions_pre_calc[self.time_step+10]
			except:
				pass
			self.action_vec = [self.current_actions[0], self.current_actions[1]]
			self.xela_vec = np.array(sync_data.data[:48])[np.newaxis, :]
			self.xela_vec = self.remove_offset(self.xela_vec)

			self.kalman_filter_func()

			self.xela_data[self.time_step, : ] = np.concatenate((self.xela_vec[0], self.xela_vec[1], self.xela_vec[2]))
			self.robot_data[self.time_step, : ]  = self.state_vec
			self.action_data[self.time_step, : ] = self.action_vec
					
		self.time_step +=1
	
	def compute_trajectory(self):
		xela_seq   = self.xela_kalman_mean[self.time_step-10 : self.time_step , : ]
		robot_seq  = self.robot_data[self.time_step-10 : self.time_step , : ]
		action_seq = self.action_data[self.time_step-10 : self.time_step , : ]
		
		self.scaled_tactile, self.scaled_action = self.scale_data(xela_seq, action_seq) # shape 10x1x48 & 20x1x2
		
		# Predictor
		pred_ort_inputs = {self.class_ort_session.get_inputs()[0].name: self.scaled_tactile, self.class_ort_session.get_inputs()[1].name: self.scaled_action}
		self.prediction = self.class_ort_session.run(None, pred_ort_inputs)
		slip_class = expit(self.prediction[0][0][0])
		slip_pred_tag = np.round(slip_class)
		self.predicted_slip_class[self.time_step] = slip_pred_tag

		self.add_to_save()

		slip_msg = Float64()
		slip_msg.data = slip_pred_tag
		self.slip_prediction_pub.publish(slip_msg)

		if self.post_slip_lin_acc_T != 0:
			self.optimal_weight = self.gen_opt_traj(action_seq[:, 0])
			self.optimal_weights[self.time_step] = self.optimal_weight
			self.optimal_trajectory[self.time_step, :, 0] = np.matmul(self.Phi, self.optimal_weight)
			self.optimal_trajectory[self.time_step, :, 1] = (self.translation_y/self.translation_x)*self.optimal_trajectory[self.time_step, 1, 0]
			
			traj_msg = Float64MultiArray()
			traj_msg.data = [self.optimal_trajectory[self.time_step, 1, 0], (self.translation_y/self.translation_x)*self.optimal_trajectory[self.time_step, 1, 0]]
			self.optimal_traj_pub.publish(traj_msg)
			self.time_step_after_slip += 1
			# print("opt traj: ", self.optimal_trajectory[self.time_step, 1, 0])
		
		self.prev_time_step = self.time_step

	def control_loop(self):

		rate = rospy.Rate(60)
		
		while not rospy.is_shutdown():
			
			try:
				if self.time_step == 0:
					self.t0 = time.time()

				if self.stop == 0.0 and self.time_step > 10 and self.time_step > self.prev_time_step:
					self.compute_trajectory()
				elif self.stop == 1.0:
					[sub.sub.unregister() for sub in self.sync_subscriber]
					break

				rate.sleep()
			
			except KeyboardInterrupt:
				break
	
	def print_rate(self):
		t1 = time.time()
		rate = self.time_step / (t1-self.t0)
		print("RATE: ", rate)


if __name__ == "__main__":
	mf = RobotTest()

	mf.print_rate()
	# mf.save_results()
