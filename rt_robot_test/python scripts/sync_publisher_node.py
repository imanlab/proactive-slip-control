#! /usr/bin/env python3
import rospy
import time
import datetime
import numpy as np
import message_filters
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Float64

def sync_publisher(tactile_data, robot_state_data, slip_onset_data, marker_data):
			sync_data_vec[:48, 0] = tactile_data.data
			sync_data_vec[48:68, 0] = robot_state_data.data
			sync_data_vec[68:75, 0] = [marker_data.pose.position.x, marker_data.pose.position.y, marker_data.pose.position.z,
														marker_data.pose.orientation.x, marker_data.pose.orientation.y, marker_data.pose.orientation.z,
														marker_data.pose.orientation.w]
			sync_data_vec[75:, 0] = slip_onset_data.data

			sync_data_msg = Float64MultiArray()
			sync_data_msg.data = sync_data_vec
			sync_data_pub.publish(sync_data_msg)
		# print("hello world")

rospy.init_node('sync_publisher_node', anonymous=True, disable_signals=True)
xela_sub    = message_filters.Subscriber('/xela1_data', Float64MultiArray)
robot_sub   = message_filters.Subscriber('/robotPose', Float64MultiArray)
slip_sub   = message_filters.Subscriber('/slipData', Float64MultiArray)
marker_sub  = message_filters.Subscriber('/aruco_simple/poseStamped', PoseStamped)
subscribers = list([xela_sub, robot_sub, slip_sub, marker_sub])
sync_data_vec = np.zeros((77, 1))
ts          = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=1, slop=0.1, allow_headerless=True)
ts.registerCallback(sync_publisher)
sync_data_pub = rospy.Publisher('/sync_data', Float64MultiArray, queue_size = 1)
rate = rospy.Rate(1000)
while not rospy.is_shutdown():
	rate.sleep()