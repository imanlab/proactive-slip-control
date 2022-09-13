#! /usr/bin/env python3
# license removed for brevity
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray


class Opt_trj_pub():
    def __init__(self):
        self.traj = np.array([0.0, 0.0])
        rospy.init_node('traj_publisher', anonymous=True)
        rospy.Subscriber("/optimal_traj", Float64MultiArray, self.callback)

    def callback(self, data):
        self.traj = data.data

if __name__ == '__main__':
    trj_pb = Opt_trj_pub()
    optimal_traj_pub = rospy.Publisher('/optimal_traj_node', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        traj_msg = Float64MultiArray()
        traj_msg.data = trj_pb.traj
        optimal_traj_pub.publish(traj_msg)
        rate.sleep()
