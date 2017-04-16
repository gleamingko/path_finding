#!/usr/bin/env python

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread, Lock

import rospy
import sensor_msgs.msg

class TrajectoryPlotter(object):

    def __init__(self):
        self.x = []
        self.y = []
        self.mutex = Lock()

        rospy.Subscriber("robot/joint_states", sensor_msgs.msg.JointState, self.joint_states_callback,
                         queue_size = 1)
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(np.random.rand(10))
        self.ax.set_xlim(-2.2, 1.0)
        self.ax.set_ylim(-0.1, 2.7)

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=100)
        plt.show()
        #plt.ion()
        
    def update(self, data):
        self.mutex.acquire()
        self.line.set_xdata(deepcopy(self.x))
        self.line.set_ydata(deepcopy(self.y))
        self.mutex.release()
        return self.line,

    def joint_states_callback(self, joint_state):
        self.mutex.acquire()
        self.x.append(joint_state.position[5])
        self.y.append(joint_state.position[3])
        if len(self.x) > 3000: self.x.pop(0)
        if len(self.y) > 3000: self.y.pop(0)
        self.mutex.release()

if __name__ == '__main__':
    rospy.init_node('move_arm_plot', anonymous=True)
    tp = TrajectoryPlotter()
    rospy.spin()
