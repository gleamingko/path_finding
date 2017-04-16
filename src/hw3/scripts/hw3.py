#!/usr/bin/env python

from copy import deepcopy
import math
import numpy
import random
from threading import Thread, Lock
import sys
import matplotlib.pyplot as plt

import actionlib
import control_msgs.msg
import geometry_msgs.msg
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import sensor_msgs.msg
import tf
import trajectory_msgs.msg
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker

def convert_to_message(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]        
    return t

def convert_from_message(msg):
    R = tf.transformations.quaternion_matrix((msg.orientation.x,
                                              msg.orientation.y,
                                              msg.orientation.z,
                                              msg.orientation.w))
    T = tf.transformations.translation_matrix((msg.position.x, 
                                               msg.position.y, 
                                               msg.position.z))
    return numpy.dot(T,R)

class RRTNode(object):
    def __init__(self):
        self.q=numpy.zeros(7)
        self.parent = None

class MoveArm(object):

    def __init__(self):
        print "HW3 initializing..."
        # Prepare the mutex for synchronization
        self.mutex = Lock()

        # min and max joint values are not read in Python urdf, so we must hard-code them here
        self.q_min = []
        self.q_max = []
        self.q_min.append(-1.700);self.q_max.append(1.700)
        self.q_min.append(-2.147);self.q_max.append(1.047)
        self.q_min.append(-3.054);self.q_max.append(3.054)
        self.q_min.append(-0.050);self.q_max.append(2.618)
        self.q_min.append(-3.059);self.q_max.append(3.059)
        self.q_min.append(-1.570);self.q_max.append(2.094)
        self.q_min.append(-3.059);self.q_max.append(3.059)

        # Subscribes to information about what the current joint values are.
        rospy.Subscriber("robot/joint_states", sensor_msgs.msg.JointState, self.joint_states_callback)

        # Initialize variables
        self.q_current = []
        self.joint_state = sensor_msgs.msg.JointState()

        # Create interactive marker
        self.init_marker()

        # Connect to trajectory execution action
        self.trajectory_client = actionlib.SimpleActionClient('/robot/limb/left/follow_joint_trajectory', 
                                                              control_msgs.msg.FollowJointTrajectoryAction)
        self.trajectory_client.wait_for_server()
        print "Joint trajectory client connected"

        # Wait for moveit IK service
        rospy.wait_for_service("compute_ik")
        self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
        print "IK service ready"

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
                                                      moveit_msgs.srv.GetStateValidity)
        print "State validity service ready"

        # Initialize MoveIt
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm") 
        print "MoveIt! interface ready"

        # How finely to sample each joint
        self.q_sample = [0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 0.4]
        self.joint_names = ["left_s0", "left_s1",
                            "left_e0", "left_e1",
                            "left_w0", "left_w1","left_w2"]

        # Options
        self.subsample_trajectory = True
        self.spline_timing = True
        self.show_plots = False

        print "Initialization done."


    def control_marker_feedback(self, feedback):
        pass

    def get_joint_val(self, joint_state, name):
        if name not in joint_state.name:
            print "ERROR: joint name not found"
            return 0
        i = joint_state.name.index(name)
        return joint_state.position[i]

    def set_joint_val(self, joint_state, q, name):
        if name not in joint_state.name:
            print "ERROR: joint name not found"
        i = joint_state.name.index(name)
        joint_state.position[i] = q

    """ Given a complete joint_state data structure, this function finds the values for 
    a particular set of joints in a particular order (in our case, the left arm joints ordered
    from proximal to distal) and returns a list q[] containing just those values.
    """
    def q_from_joint_state(self, joint_state):
        q = []
        q.append(self.get_joint_val(joint_state, "left_s0"))
        q.append(self.get_joint_val(joint_state, "left_s1"))
        q.append(self.get_joint_val(joint_state, "left_e0"))
        q.append(self.get_joint_val(joint_state, "left_e1"))
        q.append(self.get_joint_val(joint_state, "left_w0"))
        q.append(self.get_joint_val(joint_state, "left_w1"))
        q.append(self.get_joint_val(joint_state, "left_w2"))
        return q

    """ Given a list q[] of joint values and an already populated joint_state, this function assumes 
    that the passed in values are for a particular set of joints in a particular order (in our case,
    the left arm joints ordered from proximal to distal) and edits the joint_state data structure to
    set the values to the ones passed in.
    """
    def joint_state_from_q(self, joint_state, q):
        self.set_joint_val(joint_state, q[0], "left_s0")
        self.set_joint_val(joint_state, q[1], "left_s1")
        self.set_joint_val(joint_state, q[2], "left_e0")
        self.set_joint_val(joint_state, q[3], "left_e1")
        self.set_joint_val(joint_state, q[4], "left_w0")
        self.set_joint_val(joint_state, q[5], "left_w1")
        self.set_joint_val(joint_state, q[6], "left_w2")        

    """ Creates simple timing information for a trajectory, where each point has velocity
    and acceleration 0 for all joints, and all segments take the same amount of time
    to execute.
    """
    def compute_simple_timing(self, q_list, time_per_segment):
        v_list = [numpy.zeros(7) for i in range(0,len(q_list))]
        a_list = [numpy.zeros(7) for i in range(0,len(q_list))]
        t = [i*time_per_segment for i in range(0,len(q_list))]
        return v_list, a_list, t

    """ This function will perform IK for a given transform T of the end-effector. It returs a list q[]
    of 7 values, which are the result positions for the 7 joints of the left arm, ordered from proximal
    to distal. If no IK solution is found, it returns an empy list.
    """
    def IK(self, T_goal):
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = "left_arm"
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state = self.joint_state
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = "base"
        req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rospy.Duration(3.0)
        res = self.ik_service(req)
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = self.q_from_joint_state(res.solution.joint_state)
        return q

    """ This function checks if a set of joint angles q[] creates a valid state, or one that is free
    of collisions. The values in q[] are assumed to be values for the joints of the left arm, ordered
    from proximal to distal. 
    """
    def is_state_valid(self, q):
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = "left_arm"
        current_joint_state = deepcopy(self.joint_state)
        current_joint_state.position = list(current_joint_state.position)
        self.joint_state_from_q(current_joint_state, q)
        req.robot_state = moveit_msgs.msg.RobotState()
        req.robot_state.joint_state = current_joint_state
        res = self.state_valid_service(req)
        return res.valid

    # This function will plot the position, velocity and acceleration of a joint
    # based on the polynomial coefficients of each segment that makes up the 
    # trajectory.
    # Arguments:
    # - num_segments: the number of segments in the trajectory
    # - coefficients: the coefficients of a cubic polynomial for each segment, arranged
    #   as follows [a_1, b_1, c_1, d_1, ..., a_n, b_n, c_n, d_n], where n is the number
    #   of segments
    # - time_per_segment: the time (in seconds) allocated to each segment.
    # This function will display three plots. Execution will continue only after all 
    # plot windows have been closed.
    def plot_trajectory(self, num_segments, coeffs, time_per_segment):
        resolution = 1.0e-2
        assert(num_segments*4 == len(coeffs))
        t_vec = []
        q_vec = []
        a_vec = []
        v_vec = []
        for i in range(0,num_segments):
            t=0
            while t<time_per_segment:
                q,a,v = self.sample_polynomial(coeffs,i,t)
                t_vec.append(t+i*time_per_segment)
                q_vec.append(q)
                a_vec.append(a)
                v_vec.append(v)
                t = t+resolution
        self.plot_series(t_vec,q_vec,"Position")
        self.plot_series(t_vec,v_vec,"Velocity")
        self.plot_series(t_vec,a_vec,"Acceleration")
        plt.show()

    """ This is the main function to be filled in for HW3.
    Parameters:
    - q_start: the start configuration for the arm
    - q_goal: the goal configuration for the arm
    - q_min and q_max: the min and max values for all the joints in the arm.
    All the above parameters are arrays. Each will have 7 elements, one for each joint in the arm.
    These values correspond to the joints of the arm ordered from proximal (closer to the body) to 
    distal (further from the body). 

    The function must return a trajectory as a tuple (q_list,v_list,a_list,t).
    If the trajectory has n points, then q_list, v_list and a_list must all have n entries. Each
    entry must be an array of size 7, specifying the position, velocity and acceleration for each joint.

    For example, the i-th point of the trajectory is defined by:
    - q_list[i]: an array of 7 numbers specifying position for all joints at trajectory point i
    - v_list[i]: an array of 7 numbers specifying velocity for all joints at trajectory point i
    - a_list[i]: an array of 7 numbers specifying acceleration for all joints at trajectory point i
    Note that q_list, v_list and a_list are all lists of arrays. 
    For example, q_list[i][j] will be the position of the j-th joint (0<j<7) at trajectory point i 
    (0 < i < n).

    For example, a trajectory with just 2 points, starting from all joints at position 0 and 
    ending with all joints at position 1, might look like this:

    q_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([1, 1, 1, 1, 1, 1, 1]) ]
    v_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([0, 0, 0, 0, 0, 0, 0]) ]
    a_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([0, 0, 0, 0, 0, 0, 0]) ]
             
    Note that the trajectory should always begin from the current configuration of the robot.
    Hence, the first entry in q_list should always be equal to q_start. 

    In addition, t must be a list with n entries (where n is the number of points in the trajectory).
    For the i-th trajectory point, t[i] must specify when this point should be reached, relative to
    the start of the trajectory. As a result t[0] should always be 0. For the previous example, if we
    want the second point to be reached 10 seconds after starting the trajectory, we can use:

    t=[0,10]

    When you are done computing all of these, return them using

    return q_list,v_list,a_list,t

    In addition, you can use the function self.is_state_valid(q_test) to test if the joint positions 
    in a given array q_test create a valid (collision-free) state. q_test will be expected to 
    contain 7 elements, each representing a joint position, in the same order as q_start and q_goal.
    """               
    def motion_plan(self, q_start, q_goal, q_min, q_max):
        # ---------------- replace this with your code ------------------
        # This simple example creates a trajectory with just the start and goal points
        q_list = []
        q_list.append(q_start)
        q_list.append(q_goal)
        print "Example q_list:"
        print q_list
        # A provided convenience function creates the velocity and acceleration data, 
        # assuming 0 velocity and acceleration at each intermediate point, and 10 seconds
        # for each trajectory segment.
        v_list,a_list,t = self.compute_simple_timing(q_list, 10)
        print "Example v_list and a_list:"
        print v_list
        print a_list
        print "Example t:"
        print t
        return q_list, v_list, a_list, t
        # ---------------------------------------------------------------

    def project_plan(self, q_start, q_goal, q_min, q_max):
        q_list, v_list, a_list, t = self.motion_plan(q_start, q_goal, q_min, q_max)
        joint_trajectory = self.create_trajectory(q_list, v_list, a_list, t)
        return joint_trajectory

    def moveit_plan(self, q_start, q_goal, q_min, q_max):
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(q_goal)
        plan=self.group.plan()
        joint_trajectory = plan.joint_trajectory
        for i in range(0,len(joint_trajectory.points)):
            joint_trajectory.points[i].time_from_start = \
              rospy.Duration(joint_trajectory.points[i].time_from_start)
        return joint_trajectory        

    def create_trajectory(self, q_list, v_list, a_list, t):
        joint_trajectory = trajectory_msgs.msg.JointTrajectory()
        for i in range(0, len(q_list)):
            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = list(q_list[i])
            point.velocities = list(v_list[i])
            point.accelerations = list(a_list[i])
            point.time_from_start = rospy.Duration(t[i])
            joint_trajectory.points.append(point)
        joint_trajectory.joint_names = self.joint_names
        return joint_trajectory

    def execute(self, joint_trajectory):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory = joint_trajectory
        goal.goal_time_tolerance = rospy.Duration(0.0)
        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result()

    def sample_polynomial(self, coeffs, i, T):
        q = coeffs[4*i+0]*T*T*T + coeffs[4*i+1]*T*T + coeffs[4*i+2]*T + coeffs[4*i+3]
        v = coeffs[4*i+0]*3*T*T + coeffs[4*i+1]*2*T + coeffs[4*i+2]
        a = coeffs[4*i+0]*6*T   + coeffs[4*i+1]*2
        return (q,a,v)

    def plot_series(self, t_vec, y_vec, title):
        fig, ax = plt.subplots()
        line, = ax.plot(numpy.random.rand(10))
        ax.set_xlim(0, t_vec[-1])
        ax.set_ylim(min(y_vec),max(y_vec))
        line.set_xdata(deepcopy(t_vec))
        line.set_ydata(deepcopy(y_vec))
        fig.suptitle(title)

    def move_arm_cb(self, feedback):
        print 'Moving the arm'
        self.mutex.acquire()
        q_start = self.q_current
        T = convert_from_message(feedback.pose)
        print "Solving IK"
        q_goal = self.IK(T)
        if len(q_goal)==0:
            print "IK failed, aborting"
            self.mutex.release()
            return

        print "IK solved, planning"
        q_start = numpy.array(self.q_from_joint_state(self.joint_state))
        trajectory = self.project_plan(q_start, q_goal, self.q_min, self.q_max)
        if not trajectory.points:
            print "Motion plan failed, aborting"
        else:
            print "Trajectory received with " + str(len(trajectory.points)) + " points"
            self.execute(trajectory)
        self.mutex.release()

    def no_obs_cb(self, feedback):
        print 'Removing all obstacles'
        self.scene.remove_world_object("obs1")
        self.scene.remove_world_object("obs2")
        self.scene.remove_world_object("obs3")
        self.scene.remove_world_object("obs4")

    def simple_obs_cb(self, feedback):
        print 'Adding simple obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)

        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.5, 0.5, 0)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,1))

    def complex_obs_cb(self, feedback):
        print 'Adding hard obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.5, 0.2)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.6)) )
        self.scene.add_box("obs2", pose_stamped,(0.1,0.5,0.1))

    def super_obs_cb(self, feedback):
        print 'Adding super hard obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.5, 0.2)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.6)) )
        self.scene.add_box("obs2", pose_stamped,(0.1,0.5,0.1))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.0, 0.2)) )
        self.scene.add_box("obs3", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.1)) )
        self.scene.add_box("obs4", pose_stamped,(0.1,0.5,0.1))


    def plot_cb(self,feedback):
        handle = feedback.menu_entry_id
        state = self.menu_handler.getCheckState( handle )
        if state == MenuHandler.CHECKED: 
            self.show_plots = False
            print "Not showing plots"
            self.menu_handler.setCheckState( handle, MenuHandler.UNCHECKED )
        else:
            self.show_plots = True
            print "Showing plots"
            self.menu_handler.setCheckState( handle, MenuHandler.CHECKED )
        self.menu_handler.reApply(self.server)
        self.server.applyChanges()
        
    def joint_states_callback(self, joint_state):
        self.mutex.acquire()
        self.q_current = joint_state.position
        self.joint_state = joint_state
        self.mutex.release()

    def init_marker(self):

        self.server = InteractiveMarkerServer("control_markers")

        control_marker = InteractiveMarker()
        control_marker.header.frame_id = "/base"
        control_marker.name = "move_arm_marker"

        move_control = InteractiveMarkerControl()
        move_control.name = "move_x"
        move_control.orientation.w = 1
        move_control.orientation.x = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "move_y"
        move_control.orientation.w = 1
        move_control.orientation.y = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "move_z"
        move_control.orientation.w = 1
        move_control.orientation.z = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)

        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_x"
        move_control.orientation.w = 1
        move_control.orientation.x = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_y"
        move_control.orientation.w = 1
        move_control.orientation.z = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_z"
        move_control.orientation.w = 1
        move_control.orientation.y = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)

        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.BUTTON
        menu_control.always_visible = True
        box = Marker()        
        box.type = Marker.CUBE
        box.scale.x = 0.15
        box.scale.y = 0.03
        box.scale.z = 0.03
        box.color.r = 0.5
        box.color.g = 0.5
        box.color.b = 0.5
        box.color.a = 1.0
        menu_control.markers.append(box)
        box2 = deepcopy(box)
        box2.scale.x = 0.03
        box2.scale.z = 0.1
        box2.pose.position.z=0.05
        menu_control.markers.append(box2)
        control_marker.controls.append(menu_control)

        control_marker.scale = 0.25        
        self.server.insert(control_marker, self.control_marker_feedback)

        self.menu_handler = MenuHandler()
        self.menu_handler.insert("Move Arm", callback=self.move_arm_cb)
        obs_entry = self.menu_handler.insert("Obstacles")
        self.menu_handler.insert("No Obstacle", callback=self.no_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Simple Obstacle", callback=self.simple_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Hard Obstacle", callback=self.complex_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Super-hard Obstacle", callback=self.super_obs_cb, parent=obs_entry)
        options_entry = self.menu_handler.insert("Options")
        self.plot_entry = self.menu_handler.insert("Plot trajectory", parent=options_entry,
                                                     callback = self.plot_cb)
        self.menu_handler.setCheckState(self.plot_entry, MenuHandler.UNCHECKED)
        self.menu_handler.apply(self.server, "move_arm_marker",)

        self.server.applyChanges()

        Ttrans = tf.transformations.translation_matrix((0.6,0.2,0.2))
        Rtrans = tf.transformations.rotation_matrix(3.14159,(1,0,0))
        self.server.setPose("move_arm_marker", convert_to_message(numpy.dot(Ttrans,Rtrans)))
        self.server.applyChanges()


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm', anonymous=True)
    ma = MoveArm()
    rospy.spin()

