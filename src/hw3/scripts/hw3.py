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

import datetime


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
    #-----------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------- 
    # John Sy, Nate Apgar, Brian Bradley
    # MECE E4602, Fall 2015  

    # Motion Planning Strategy boolean. 
    # Set to True to use PRM and False to use RRT
    global usePRM
    usePRM = True

    # Computes unit directional vector
    def getUnitDirVec(self,start,end):
	direction = numpy.subtract(end,start)
	unitDirVec = direction/numpy.linalg.norm(direction)
	return unitDirVec

    #
    '''## Shortcutting'''
    # This function creates shortcuts the path to the goal in order to achieve 
    # the most direct route possible from the current nodes.   
    def shortcut(self,q_list):
        i = 0
        while i < len(q_list)-2:
            unit_dirVec = self.getUnitDirVec(q_list[i],q_list[i+2])
            step = 0.1
            valid_segment = True
            while valid_segment and numpy.linalg.norm(numpy.subtract((q_list[i] + step*unit_dirVec),q_list[i+2])) > 0.1:
                if self.is_state_valid(q_list[i] + step*unit_dirVec):
                   step = step + 0.1
                else:
                   valid_segment = False
            if valid_segment:
                del q_list[i+1]
            else:
                i = i + 1
        return q_list


    #	
    '''## Re-sampling'''
    # This function resamples the q_list so that the segments are spaced apart as 
    # closed together as possible while still being 0.5 units apart
    def resample(self,q_list):
        i = 0
	while i < len(q_list)-1:
	    # If the distance between two nodes is greater than 1, insert segments
            if numpy.linalg.norm(numpy.subtract(q_list[i+1],q_list[i])) >= 1:
                unit_dirVec = self.getUnitDirVec(q_list[i],q_list[i+1])
                dirVec = numpy.subtract(q_list[i+1],q_list[i])
                numSegmentInsert = int(numpy.linalg.norm(dirVec)/0.5) 
                # Round down so that the segments are bigger than 0.5

                segmentLength = numpy.linalg.norm(dirVec)/numSegmentInsert

	        for j in range(numSegmentInsert-1): # number of nodes = number of segments - 1
                    q_list.insert(i+j+1,q_list[i]+segmentLength*(j+1)*unit_dirVec)
                i = i + numSegmentInsert
            else: 
                i += 1
        return q_list
    
    #
    '''## Trajectory Execution'''
    # This function takes in q_list and outputs a_list and v_list as well as the spline 
    # coefficients a, b, c, and d
    def traj_compute(self,q_list):
            N = len(q_list) # N = number of waypoints
            v_list = numpy.empty((N, 7))
            a_list = numpy.empty((N, 7))
            a = numpy.empty((N-1, 7))
            b = numpy.empty((N-1, 7))
            c = numpy.empty((N-1, 7))
            d = numpy.empty((N-1, 7))    

            # coefficients = [a1, b1, c1, d1, a2, b2, c2, ....., an, bn, cn, dn] - for each joint
            coefficients = numpy.zeros((4*(N-1), 7))
    
            for joint in range(0, 7):
                A = numpy.zeros((4*(N-1), 4*(N-1)))
                b = numpy.zeros((4*(N-1)))
        
                # qdot0 and qdotfinal = 0
		# Sets our two constraints in order to have a fully determined system
                A[0][2] = 1
                A[1][4*(N-1) - 4] = 3
                A[1][4*(N-1) - 3] = 2
                A[1][4*(N-1) - 2] = 1
            
                row = 2
        
                # d_i = q_i
		# Solves the d coefficient for each segment
                for i in range(0, N-1):
                    A[row][4*i + 3] = 1 
                    b[row] = q_list[i][joint]
                    row += 1
        
                # a_i + b_i + c_i + d_i = q_i+1
		# Solves the velocity of the segment at the next time step
                for i in range(0, N-1):
                    A[row][4*i:4*i+4] = 1
                    b[row] = q_list[i + 1][joint]
                    row += 1
        
                # qdotT_i - qdot0_i+1= 0 (continuous velocity)
		# Sets the initial velocity of a segment equal to the final velocity 
		# of the previous segment
                for i in range(0, N-2):
                    A[row][4 * i] = 3
                    A[row][4 * i + 1] = 2
                    A[row][4 * i + 2] = 1
                    A[row][4 * i + 6] = -1
                    row += 1
            
                # qddot_i - qddot_i+1 = 0 (continuous acceleration)
		# Sets the initial acceleration of a segment equal to the final velocity 
		# of the previous segment
                for i in range(0, N-2):
                    A[row][4 * i] = 6
                    A[row][4 * i + 1] = 2
                    A[row][4 * i + 5] = -2
                    row += 1 
		    
		# Solve for the coefficients of the current joint
                current_coefficients = numpy.dot(numpy.linalg.inv(A), b)
		 
                for i in range(0, len(A)):
                    coefficients[i][joint] = current_coefficients[i]
            
                #use coefficients to generate v_list and a_list
                for i in range(0, N-1):
                    v_list[i][joint] = coefficients[4*i + 2][joint]
                    a_list[i][joint] = 2*coefficients[4*i + 1][joint]
                
                v_list[N-1][joint] = 0
                a_list[N-1][joint] = 6*current_coefficients[0] + 2*current_coefficients[1]
		
	        t = numpy.empty(len(q_list))
		for i in range(0,len(q_list)):
	            t[i] = i
            
	    # Show plots
	    if self.show_plots != False:
	        self.plot_trajectory(len(q_list)-1,coefficients[:,1],1)
	
            return v_list, a_list, t, coefficients


    # Performs Dijkstra's algorithm
    def Dijkstra(self,startNode,roadmap,goalNode):
        keepMoving = []
        superRoadMap = []
        superRoadMap.append(startNode)
        for i in range(len(roadmap)):
            superRoadMap.append(roadmap[i])
        superRoadMap.append(goalNode)
                
        # Now superRoadMap contains all the nodes, including start and goal
        # startNode, then a bunch of random nodes, and lastly goalNode
        for i in range(len(superRoadMap)):
            superRoadMap[i].parent.append([-1,False]) # The number is the g(n) and the boolean determines if it has been visited
            keepMoving.append(False)
        startNode.parent[-1][0] = True
        # Now all the nodes have [-1,False] at the end of their parentList except startNode [0,False]
        # This node info can be accessed by superRoadMap[i].parent[-1][0 or 1]
        print keepMoving
        while not all(keepMoving):
	    #print "loop 0"
            #print "Rudolph"
            lowestPathIndex = 0
            lowestPathLength = 10000000000000
            for i in range(len(superRoadMap)): # Find the node with lowest path length
                if superRoadMap[i].parent[-1][0] > 0 and superRoadMap[i].parent[-1][0] <= lowestPathLength and superRoadMap[i].parent[-1][1] == False:
                    lowestPathIndex = i
                    lowestPathLength = superRoadMap[i].parent[-1][0]
            # Sometimes when lowestPathLength == 10000000000000, which means it somehow did not pick up the node with lowest g(n), it goes into infinite loop
            print lowestPathIndex
            print lowestPathLength
            #print "Lowest Path Index: " + str(lowestPathIndex)
            superRoadMap[lowestPathIndex].parent[-1][1] = True # Mark it as visited
            for i in range(len(superRoadMap[lowestPathIndex].parent)-1): # -1 because don't want to visit the last item that contains node info
                if superRoadMap[lowestPathIndex].parent[i].parent[-1][1] == False:
                    d = numpy.linalg.norm(numpy.subtract(superRoadMap[lowestPathIndex].q,superRoadMap[lowestPathIndex].parent[i].q))
                    if superRoadMap[lowestPathIndex].parent[i].parent[-1][0] == -1:
                        superRoadMap[lowestPathIndex].parent[i].parent[-1][0] = superRoadMap[lowestPathIndex].parent[-1][0] + d
                    else:
                        superRoadMap[lowestPathIndex].parent[i].parent[-1][0] = min(superRoadMap[lowestPathIndex].parent[-1][0] + d, superRoadMap[lowestPathIndex].parent[i].parent[-1][0])
            for i in range(len(superRoadMap)):
                keepMoving[i] = superRoadMap[i].parent[-1][1]
            print keepMoving
        # Now all nodes in superRoadMapshould have g(n)
        return superRoadMap


    # Finds the shortest path from the start to the goal and the shortest distance
    def DistanceT(self,superRoadMap):
        rev_shortestPath = [] # This will contain the nodes of the shortest path from goal to start
        rev_shortestPath.append(superRoadMap[-1])
        shortestPath = [] # This will contain the coordinates of the shortest path from start to goal
        shortestPath.append(superRoadMap[0].q)
        shortestLength = 0
        currentNode = superRoadMap[-1]
        while currentNode != superRoadMap[0]:
	    #print "loop 1"
            smallestGD = 1000000000
            i_smallestGD = 0
            for i in range(len(currentNode.parent)-1):
                GD = currentNode.parent[i].parent[-1][0] + numpy.linalg.norm(numpy.subtract(currentNode.q,currentNode.parent[i].q))
                if GD < smallestGD:
                    smallestGD = GD
                    i_smallestGD = i
            rev_shortestPath.append(currentNode.parent[i_smallestGD])
            currentNode = currentNode.parent[i_smallestGD]
            shortestLength = shortestLength + smallestGD
        for i in range(len(rev_shortestPath)-1,-1,-1):
            shortestPath.append(rev_shortestPath[i].q)
        for i in range(len(superRoadMap)):
            del superRoadMap[i].parent[-1] # Clean up the node info
        return shortestPath, shortestLength



    # Conduct motion planning using PRM
    def PRM(self, q_start, q_goal, q_min, q_max):
	q_list = []
        q_list.append(q_start)

	# This will contain all nodes (not q's) to be inserted
        nodes = [] 

	# This will contain all coordinates to be inserted in reverse order
        rev_q_list = [] 

	roadmap = []
        startNode = RRTNode()
        startNode.q = q_start
        startNode.parent = []
        goalNode = RRTNode()
        goalNode.q = q_goal
        goalNode.parent = []
        start_goal = []
        start_goal.append(startNode)
        start_goal.append(goalNode)
        T = 60 # in seconds
        start_time = datetime.datetime.now()
        temp_q_list = []
        plot_length = []
        previous_time_stamp = datetime.datetime.now()

        while (datetime.datetime.now()-start_time).seconds < T:
	    #print "loop 2"
            valid_r = False # Set false first just to get into the loop
            r = []
            for i in range(len(q_min)):
                r.append((q_max[i]-q_min[i])*numpy.random.random_sample()+q_min[i])
            if self.is_state_valid(r):
                newNode = RRTNode()
                newNode.q = r
                newNode.parent = []
                roadmap.append(newNode)
                # Check connection to other nodes in roadmap
                for i in range(len(roadmap)-1): # Excluding the newest node added
                    unitDir = self.getUnitDirVec(roadmap[i].q,newNode.q)
                    step = 0.1
                    valid_segment = True
                    while valid_segment and numpy.linalg.norm(numpy.subtract((roadmap[i].q + step*unitDir),newNode.q)) > 0.1:
			#print "Loop 3"
                        if self.is_state_valid(roadmap[i].q + step*unitDir):
                            step = step + 0.1
                        else:
                            valid_segment = False
                    if valid_segment: # Connect the two points together
                        roadmap[i].parent.append(newNode)
                        newNode.parent.append(roadmap[i])
                # Check connection to start and goal node
                for i in range(len(start_goal)):
                    unitDir = self.getUnitDirVec(start_goal[i].q,newNode.q)
                    step = 0.1
                    valid_segment = True
                    while valid_segment and numpy.linalg.norm(numpy.subtract((start_goal[i].q + step*unitDir),newNode.q)) > 0.1:
			#print "loop 4"
                        if self.is_state_valid(start_goal[i].q + step*unitDir):
                            step = step + 0.1
                        else:
                            valid_segment = False
                    if valid_segment: # Connect the start/goal point with newNode
                        start_goal[i].parent.append(newNode)
                        newNode.parent.append(start_goal[i])
                if len(newNode.parent) == 0:
                    del roadmap[-1]
                if len(startNode.parent) > 0 and len(goalNode.parent) > 0:
                    temp_q_list, temp_distance = self.DistanceT(self.Dijkstra(startNode,roadmap,goalNode))
                    if (datetime.datetime.now()-previous_time_stamp).seconds >= 1 and (datetime.datetime.now()-previous_time_stamp).microseconds >= 1:
                        previous_time_stamp = datetime.datetime.now()
                        plot_length.append(temp_distance)
        if len(startNode.parent) == 0 or len(goalNode.parent) == 0:
	    print "No path found after " + str(T) + " seconds"
            return []

        plot_time = range(len(plot_length))
        self.plot_series(plot_time,plot_length,'Shortest Path by Dijkstra')
        plt.show() # Uncomment this if you want to see the plot, but need to close it before the robot moves

        q_list = temp_q_list

        return q_list



    # Conduct motion planning using RRT
    def RRT(self, q_start, q_goal, q_min, q_max):
	q_list = []
        q_list.append(q_start)

	# This will contain all nodes (not q's) to be inserted
        nodes = [] 

	# This will contain all coordinates to be inserted in reverse order
        rev_q_list = [] 

        # Find a valid random point within legal range
        q_see_goal = False
        while (q_see_goal == False) and len(nodes) <= 2000:
	    # Set false first just to intialize the loop
            valid_r = False 
            r = []
            for i in range(len(q_min)):
                r.append((q_max[i]-q_min[i])*numpy.random.random_sample()+q_min[i])

            # Find the point p in tree closest to r        
            closest_node = RRTNode()
	    # Temporarily make q_start to be the closest node
            closest_node.q = q_start 
            closest_node.parent = None
            closest_node_distance = numpy.linalg.norm(numpy.subtract(closest_node.q,r))
            for p in nodes:
		# Compute p-r distance
                p_r_distance = numpy.linalg.norm(numpy.subtract(p.q,r)) 
                if p_r_distance < closest_node_distance:
		    # now closest_node is the referring to p
                    closest_node = p 
                    closest_node_distance = p_r_distance

            '''# Now closest_node holds the parent node for the potential new node 
	    # (not instantiated yet)
            # Think of each name as individual pointer; can point to the same 
	    # var/object or be reassigned'''

            # Compute the direction of growth from closest_node
            unit_dir = self.getUnitDirVec(closest_node.q,r)
            r_insert = closest_node.q + 0.5*unit_dir # Potential insertion point coordinates
            

            # Now instantiate the potential node
            newNode = RRTNode()
            newNode.q = r_insert
            newNode.parent = closest_node
        
            # Check if the segment is valid before insertion 
	    # (else skip the rest and find another point)
	    # Only sample the path segment if the potential node is valid
            if self.is_state_valid(newNode.q): 
                step = 0.1
                valid_segment = True # Just to initialize loop
                while valid_segment and step < 0.5:
                    if self.is_state_valid(closest_node.q + step*unit_dir):
                        step = step + 0.1
                    else:
                        valid_segment = False
		# Add new node to the tree if the segment is valid 
		# (else skip the rest and find another point)
                if valid_segment: 
                    nodes.append(newNode)
                    print "Number of nodes in tree: " + str(len(nodes))
                    # Check if it can connect to the goal after the new node is
		    # added (else skip the rest and find another point)
                    unit_dir_to_goal = self.getUnitDirVec(newNode.q,q_goal)
                    step = 0.1
                    valid_segment = True # Redefined for checking newNode to goal
                    while valid_segment and numpy.linalg.norm(numpy.subtract((newNode.q + step*unit_dir_to_goal),q_goal)) > 0.1:
                        if self.is_state_valid(newNode.q + step*unit_dir_to_goal):
                            step = step + 0.1
                        else:
                            valid_segment = False
		    # valid_segment==True, then stop getting new nodes; 
		    # else loop back again
                    if valid_segment: 
                        q_see_goal = True
                        currentNode = nodes[len(nodes)-1]
                        #print numpy.subtract(currentNode.q,q_start)
                        #print all(numpy.subtract(currentNode.q,q_start))
                        while all(numpy.subtract(currentNode.q,q_start)):
                            rev_q_list.append(currentNode.q)
                            currentNode = currentNode.parent
        '# End of while loop for getting new nodes'           

        # Quit trying if too much effort
        if q_see_goal == False:
	    print "Error: Goal not found."
            return [], [], [], 0

        # Push the path nodes to q_list
        for i in range(len(rev_q_list)-1,-1,-1):
            q_list.append(rev_q_list[i])
        q_list.append(q_goal) 
	'# Now q_list contains all the path nodes'

        return q_list



    # --------------------------------------
    '''## Main motion planning funtion'''
    # --------------------------------------
    def motion_plan(self, q_start, q_goal, q_min, q_max):
	# Choose motion planning strategy
	if usePRM == True:
	    q_list = self.PRM(q_start, q_goal, q_min, q_max)
        else:
	    q_list = self.RRT(q_start, q_goal, q_min, q_max)
	
	if q_list == []:
	    return [], [] , [], 0

	# Shortcut the current q_list
	q_list = self.shortcut(q_list)
        
	# Resample the current q_list
	q_list = self.resample(q_list)
	    
        '# q_list is finalized; end of motion planning'

	# Trajectory Planning
        v_list,a_list,t,coeffs = self.traj_compute(q_list)

        return q_list, v_list, a_list, t

    #------------------------------------------------------------------------------
    '''# End of our HW3 code'''
    # -----------------------------------------------------------------------------

    def project_plan(self, q_start, q_goal, q_min, q_max):
        q_list, v_list, a_list, t = self.motion_plan(q_start, q_goal, q_min, q_max)
        joint_trajectory = self.create_trajectory(q_list, v_list, a_list, t)
        return joint_trajectory

    def moveit_plan(self, q_start, q_goal, q_min, q_max):
        self.group.clear_pose_tarUs()
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

