#! /usr/bin/env python3
import rospy
import json
import websocket
import time
from std_msgs.msg import Float64MultiArray


ip = "10.5.32.139"
port = 5000

save_list = []

rospy.init_node('xela_sensor')
xela1_pub = rospy.Publisher('/xela1_data', Float64MultiArray, queue_size = 1)
xela2_pub = rospy.Publisher('/xela2_data', Float64MultiArray, queue_size = 1)
rate = rospy.Rate(100) # 100hz
time.sleep(5)

def publisher(xela_pub, data):
    xela_msg = Float64MultiArray()
    xela_msg.data = data
    xela_pub.publish(xela_msg)


def on_message(wsapp, message):
    data = json.loads(message)
    # print(data)
    sensor1 = data['1']['data'].split(",")
    sensor2 = data['2']['data'].split(",")
    txls = int(len(sensor1)/3)
    data1_row = []
    data2_row = []
    for i in range(txls):
        x = int(sensor1[i*3],16)
        y = int(sensor1[i*3+1],16)
        z = int(sensor1[i*3+2],16)
        data1_row.append(float(x))
        data1_row.append(float(y))
        data1_row.append(float(z))
        x2 = int(sensor2[i*3],16)
        y2 = int(sensor2[i*3+1],16)
        z2 = int(sensor2[i*3+2],16)
        data2_row.append(float(x2))
        data2_row.append(float(y2))
        data2_row.append(float(z2))


    publisher(xela1_pub, data1_row)
    publisher(xela2_pub, data2_row)
    rate.sleep()
       

wsapp = websocket.WebSocketApp("ws://{}:{}".format(ip,port), on_message=on_message)
wsapp.run_forever()
