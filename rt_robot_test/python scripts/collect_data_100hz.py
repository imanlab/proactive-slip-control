#! /usr/bin/env python3
# import tf
import os
import sys
import time
import rospy
import cv2
import select
import datetime
import actionlib
import numpy as np
import termios, tty
import pandas as pd
import message_filters
import moveit_msgs.msg
# import moveit_commander

from scipy import signal
from sensor_msgs.msg import JointState
from pynput.keyboard import Key, Listener
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
from franka_gripper.msg import HomingAction
from franka_gripper.msg import MoveAction, MoveActionGoal, GraspAction, GraspActionGoal
from std_msgs.msg import Int16MultiArray, Float32MultiArray, Float64MultiArray


class RobotReader():
	def __init__(self):
		rospy.init_node('data_collection_client', anonymous=True, disable_signals=False)
		self.settings = termios.tcgetattr(sys.stdin)

		while input("press enter to start saving data, or type ctrl c then n to not: ") != "n":
			self.stop 					 = False
			self.cpp_flag                = 0.0
			self.xelaSensor1 			 = []
			self.xelaSensor2 			 = []
			self.robot_pose 			 = []
			self.robot_joint			 = []
			self.robot_joint_vel		 = []
			self.marker_data			 = []

			self.listener = Listener(on_press=self.start_collection)
			self.listener.start()
			print(self.stop)
			self.robot_pose_sub = message_filters.Subscriber('/robotPose', Float64MultiArray)
			self.robot_joint_sub = message_filters.Subscriber('/robotJoint', Float64MultiArray)
			self.robot_joint_vel_sub = message_filters.Subscriber('/robotJointVel', Float64MultiArray)
			self.xela1_sub = message_filters.Subscriber('/xela1_data', Float64MultiArray)
			self.xela2_sub = message_filters.Subscriber('/xela2_data', Float64MultiArray)
			self.marker_sub = message_filters.Subscriber('/aruco_simple/poseStamped', PoseStamped)
			subscribers = [self.robot_joint_sub, self.robot_joint_vel_sub, self.robot_pose_sub, self.xela1_sub, self.xela2_sub, self.marker_sub]
			self.start_time = datetime.datetime.now()
			print(self.start_time)
			self.counter = 0
			self.prev_i = 0
			self.i = 1
			self.ts = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=1, slop=0.1, allow_headerless=True)
			self.ts.registerCallback(self.read_robot_data)
			rate = rospy.Rate(1000)
			while (not rospy.is_shutdown()) and (self.stop is False) and (self.cpp_flag == 0.0):
				if self.prev_i==0:
					t0 = time.time()
				self.i += 1
				rate.sleep()
			t1 = time.time()
			self.end_subscription()
			self.stop = False
			self.stop_time = datetime.datetime.now()
			print(self.stop_time)
			self.rate = (len(self.xelaSensor1)) / (t1-t0)
			print("\n Stopped the data collection \n now saving the stored data")
			self.listener.stop()
			self.save_data()


	def read_robot_data(self, joint_data, joint_vel_data, robot_pose_data, xela1_data, xela2_data, marker_data):
		if (self.stop == False) and (self.i != self.prev_i) and (self.cpp_flag == 0.0):
			self.cpp_flag = robot_pose_data.data[-1]
			self.prev_i = self.i
			self.xelaSensor1.append(xela1_data.data)
			self.xelaSensor2.append(xela2_data.data)
			self.robot_joint.append(joint_data.data)
			self.robot_joint_vel.append(joint_vel_data.data)
			self.robot_pose.append(robot_pose_data.data[:-1])
			self.marker_data.append([marker_data.pose.position.x, marker_data.pose.position.y, marker_data.pose.position.z,
									  marker_data.pose.orientation.x, marker_data.pose.orientation.y, marker_data.pose.orientation.z,
									  marker_data.pose.orientation.w,])
		

	def end_subscription(self):
		self.listener.stop()
		self.robot_pose_sub.unregister()
		self.robot_joint_sub.unregister()
		self.robot_joint_vel_sub.unregister()
		self.xela1_sub.unregister()
		self.xela2_sub.unregister()
		self.marker_sub.unregister()

	def start_collection(self, key):
		print("herer")
		if key == Key.esc:
			self.stop = True
			self.listener.stop()
			self.robot_pose_sub.unregister()
			self.robot_joint_sub.unregister()
			self.robot_joint_vel_sub.unregister()
			self.xela1_sub.unregister()
			self.xela2_sub.unregister()
			self.marker_sub.unregister()

	def save_data(self):

		self.xelaSensor1 = np.array(self.xelaSensor1)
		self.xelaSensor1 = np.reshape(self.xelaSensor1, (self.xelaSensor1.shape[0], 48))
		self.xelaSensor2 = np.array(self.xelaSensor2)
		self.xelaSensor2 = np.reshape(self.xelaSensor2, (self.xelaSensor2.shape[0], 48))
		
		self.marker_pose = np.asarray(self.marker_data)

		self.robot_joint 	 = np.asarray(self.robot_joint)
		self.robot_joint_vel = np.asarray(self.robot_joint_vel)
		self.robot_pose      = np.asarray(self.robot_pose)
		
		self.robot_data = np.concatenate((self.robot_joint, self.robot_joint_vel, self.robot_pose), axis=1)
		
		print("robot_pose_formated; ", self.robot_data.shape)
		print("xelaSensor1 length: ", self.xelaSensor1.shape)
		print("xelaSensor2 length: ", self.xelaSensor2.shape)
		print("markerFormated length: ", self.marker_pose.shape)
		print("rate: ", self.rate)

		# If the subscriber executed one last time half way through processing the data (Happens sometimes and we dont know why)
		# Then	remove last element in data trial for robot data:
		if self.marker_pose.shape[0] > self.xelaSensor1.shape[0]:
			self.marker_pose = self.marker_pose[0:-2]
			self.robot_data = self.robot_data[0:-2]

		T1 = pd.DataFrame(self.xelaSensor1)
		T2 = pd.DataFrame(self.xelaSensor2)
		T3 = pd.DataFrame(self.marker_pose)
		T4 = pd.DataFrame(self.robot_data)

		xela_Sensor_col = ['txl1_x', 'txl1_y', 'txl1_z', 'txl2_x', 'txl2_y', 'txl2_z','txl3_x', 'txl3_y', 'txl3_z','txl4_x', 'txl4_y', 'txl4_z','txl5_x', 'txl5_y', 'txl5_z','txl6_x', 'txl6_y', 'txl6_z',
		'txl7_x', 'txl7_y', 'txl7_z','txl8_x', 'txl8_y', 'txl8_z','txl9_x', 'txl9_y', 'txl9_z','txl10_x', 'txl10_y', 'txl10_z','txl11_x', 'txl11_y', 'txl11_z','txl12_x', 'txl12_y', 'txl12_z',
		'txl13_x', 'txl13_y', 'txl13_z','txl14_x', 'txl14_y', 'txl14_z','txl15_x', 'txl15_y', 'txl15_z','txl16_x', 'txl16_y', 'txl16_z']

		robot_states_col = ["position_joint1", "position_joint2", "position_joint3", "position_joint4", "position_joint5", "position_joint6", "position_joint7",
		"velocity_joint1", "velocity_joint2", "velocity_joint3", "velocity_joint4", "velocity_joint5", "velocity_joint6", "velocity_joint7",
		"ee_T_00", "ee_T_10", "ee_T_20", "ee_T_30", "ee_T_01", "ee_T_11", "ee_T_21", "ee_T_31", "ee_T_02", "ee_T_12", "ee_T_22", "ee_T_32", "ee_T_03",
		"ee_T_13", "ee_T_23", "ee_T_33", "v_x", "v_y", "time"]

		marker_pose_col = ["marker_position_x", "marker_position_y", "marker_position_z", 
						   "marker_quaternion_x", "marker_quaternion_y", "marker_quaternion_z", "marker_quaternion_w",]

		# create new folder for this experiment:
		self.folder = str('/home/kiyanoush/Cpp_ws/src/robotTest/data/Quadratic/data_sample_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
		mydir = os.mkdir(self.folder)

		T1.to_csv(self.folder + '/xela_sensor1.csv', header=xela_Sensor_col, index=False)
		T2.to_csv(self.folder + '/xela_sensor2.csv', header=xela_Sensor_col, index=False)
		T3.to_csv(self.folder + '/marker.csv', header=marker_pose_col, index=False)
		T4.to_csv(self.folder + '/robot_state.csv', header=robot_states_col, index=False)

		# Create meta data
		save = input("save meta file? 'n' to not")
		if save != 'n':
			meta_data = ['dropped', 'notes']
			meta_data_ans = [] # ["1", "0", "1", "1", "NOT KINESTHETIC"]
			for info in meta_data:
				value = input(str("please enter the " + info))
				meta_data_ans.append(value)
			meta_data.extend(('sensor_type', 'frequency_hz', 'start_time', 'stop_time'))
			meta_data_ans.extend(('xela_2fingers_glove', str(self.rate), str(self.start_time), str(self.stop_time)))
			meta_data_ans = np.array([meta_data_ans])
			T5 = pd.DataFrame(meta_data_ans)
			T5.to_csv(self.folder + '/meta_data.csv', header=meta_data, index=False)

if __name__ == "__main__":
	robot_reader = RobotReader()
