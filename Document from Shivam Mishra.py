#!/usr/bin/env python3

"""
Controller for the drone
"""

# standard imports
import copy
import time

# third-party imports
import scipy.signal
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PidTune
from swift_msgs.msg import PIDError, RCMessage
from swift_msgs.srv import CommandBool
import json
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

import sys
from loc_msg.msg import Biolocation

MIN_ROLL = 1250
BASE_ROLL = 1490
MAX_ROLL = 1600
SUM_ERROR_ROLL_LIMIT = 10000

#pitch
MIN_PITCH = 1250
BASE_PITCH = 1455
MAX_PITCH = 1600
SUM_ERROR_PITCH_LIMIT = 10000

#throttle
MIN_THROTTLE = 1250
BASE_THROTTLE = 1450
MAX_THROTTLE = 1600
SUM_ERROR_THROTTLE_LIMIT = 10000

DRONE_WHYCON_POSE = [[], [], []]

try:
    points = None
except Exception as E:
    print("No Points")
    # points = None
finally:
    pass
    


class DroneController():
    def __init__(self,node):

        self.node= node
        self.bridge = CvBridge()
        
        self.image_sub = node.create_subscription(
            Image,
            '/video_frames',
            self.image_callback,
            1  # queue_size
        )
        self.biolocation_publisher = node.create_publisher(
            Biolocation,
            '/astrobiolocation',
            1  # queue_size
        )
        self.corner = []
        self.detect_Led = True
        self.total_time = Clock().now().nanoseconds
        self.loop_start_time = 0
        self.blinked=False
        """if roof camera is 100% falt keep it zero"""
        self.whycon_bias_y = -1.447 # to adjust celing camera tilt for y (adjusted for focal length)
        self.whycon_bias_x = 0.0473 # to adjust celing camera tilt for x (adjusted for focal length)

        self.landing_path = []
        self.tolerance_list = [0.6,0.6,0.9]
        self.rc_message = RCMessage()
        self.drone_whycon_pose_array = PoseArray()
        self.last_whycon_pose_received_at = 0
        self.commandbool = CommandBool.Request()
        service_endpoint = "/swift/cmd/arming"
        self.correction_scale_x = 0.0050
        self.correction_scale_y = 0.0050
        self.landing_waypoints = []
        self.landing_waypoint_index = 0
        self.disarm = False
        self.landed = False
        # self.shm_null = None
        self.arming_service_client = self.node.create_client(CommandBool,service_endpoint)
        self.set_points = [-0, -0, 25]        # set_pointss for x, y, z respectively      
        # arena_size_x = 9  # Size of the arena in the x-axis
        # arena_size_y = 5 # Size of the arena in the y-axis
        # altitude = 21  # Altitude at which you want to hover
        # self.drone_position = [0.0, 0.0, 0.0] #-----------------------------------------------------------------------
        self.centroid_of_LEDs=[]
        self.first_time=True
        self.current_led = None
        self.current_goal_index=-1
        self.aliens=False
        self.maximum_LEDs=0
        self.time_led=0
        # self.image_error = 65
        self.image_errors_x = { 2: 100, 3:100, 4: 100, 5: 100}
        self.image_errors_y = { 2: 100, 3:100, 4: 100, 5: 100}
        self.total_leds=2
        self.believe=False
        self.task_done=False
        self.waypoint_tolerance = 0.8
        self.first_time_alien_a=True
        self.first_time_alien_b=True
        self.first_time_alien_c=True
        self.coor_alien=dict()
        self.publish_values = True
        self.waypoint_update = True
        # [x_set_points, y_set_points, z_set_points]
        # whycon marker at the position of the dummy given in the scene. Make the whycon marker associated with position_to_hold dummy renderable and make changes accordingly
        # self.set_points = [2, 2, 23]
        self.goal_of_the_match=False
        self.error_x_y=[]
        self.goal_path=[]
        self.all_clusters=[]

        self.waypoints = []
        self.blink3 = False
        self.count_started = False
        self.count_aux = 0
        self.time = 0
        self.counter = 0
        self.blink_and_beep = [False , 0]
        self.elapsed_couter_start = False
        self.LED_mili=[]
        #for small sqaure
        self.center_y = -3.8
        self.center_x = -1.5
        self.square_size = 2
        self.grid_length = 0.4
        self.y_max_pos = 6
        self.y_max_neg = 9
        self.x_max_pos = 8
        self.x_max_neg = 9
        self.z_base = 32
        #for wapoints
        self.grid_spacing_x = 0.5 # You can change this value based on your requirements
        self.grid_spacing_y = 4 # You can change this value based on your requirements

        
        self.arena_size_x = 8
        self.arena_size_y = 5
        self.arena_size_x_n = -7
        self.arena_size_y_n = -10
        self.altitude = self.z_base - 7
        # self.callback_frequency = 30
        # self.callback_count = 0
        # self.callback_timer = Clock().now().nanoseconds
        # self.callback_window = 1.0 * 1e9

        self.shm = False
        
        # self.corners = {"Bottom-Right": False, "Bottom-Left" : False}
        
        self.published_leds = []
        self.waypoints = []
        
        self.hexagon_inde = 0

        self.landing_cooridnates = [-8,-12, self.z_base]

        # Starting point (center)
        self.start_point = (0, 0)
        self.hexagon_waypoints = []

        # Number of concentric hexagons
        
# print(waypoints)
        # self.waypoints = waypoints
        self.small_square = []
        
        self.current_waypoint_index = 0
        self.original = self.waypoints
        
        self.error = [0, 0, 0]         # Error for roll, pitch and throttle     
        

        # Create variables for integral and differential error
        self.prev_error = [0.0, 0.0, 0.0] #array to store prev_recored errors in 3 cardinal directions
        self.sum_error = [0.0, 0.0, 0.0] #sum error

        # Create variables for previous error and sum_error
        self.integral = [0,0,0]
        self.derivative = [0,0,0]
        self.hexagon_pattern = []
        self.ek_aur_var = True
        self.alignment_needed = True
        # self.Kp = [ 459*0.01  , 524*0.01  , 3.93  ]
        # self.Ki = [ 95.00*0.0002  , 55*0.0002 , 0.0162 ] # Ki values for roll, pitch, and altitude
        # self.Kd = [ 1550*0.1  , 3559*0.1  , 109.2  ]

        #.............
        # self.Kp = [ 860*0.01  , 590*0.01  , 3.93  ]
        # self.Ki = [ 110.00*0.0002  , 90*0.0002 , 0.0162 ] # Ki values for roll, pitch, and altitude
        # self.Kd = [ 2336*0.1  , 3559*0.1  , 109.2  ]

        # self.Kp = [ 860*0.01  , 590*0.01  , 3.93  ]
        # self.Ki = [ 150.00*0.0002  , 120*0.0002 , 0.0192 ] # Ki values for roll, pitch, and altitude
        # self.Kd = [ 2336*0.1  , 3559*0.1  , 109.2  ]
        self.Kp = [ 850*0.01  , 600*0.01  , 3.93  ]
        # self.Ki = [ 160.00*0.0002  , 130*0.0002 , 0.0202 ] 
        # self.Ki = [ 140.00*0.0002  , 160*0.0002 , 0.0238 ]  #battery-1
        self.Ki = [ 160.00*0.0002  , 130*0.0002 , 0.0202 ]  #battery-2
        # Ki values for roll, pitch, and altitude
        self.Kd = [ 2236*0.1  , 3359*0.1  , 119.2  ]

        # Similarly create variables for Kd and Ki

        # Create subscriber for WhyCon 
        self.first_time_alien = {2 : True, 3: True, 4: True, 5: True}
        
        self.whycon_sub = node.create_subscription(PoseArray,"/whycon/poses",self.whycon_poses_callback,1)
        
        # Similarly create subscribers for pid_tuning_altitude, pid_tuning_roll, pid_tuning_pitch and any other subscriber if required
       
        self.pid_alt = node.create_subscription(PidTune,"/pid_tuning_altitude",self.pid_tune_throttle_callback,1)
        self.pid_pitch = node.create_subscription(PidTune,"/pid_tuning_pitch",self.pid_tune_pitch_callback,1)
        self.pid_roll = node.create_subscription(PidTune,"/pid_tuning_roll",self.pid_tune_roll_callback,1)
        

        # Create publisher for sending commands to drone 

        self.rc_pub = node.create_publisher(RCMessage, "/swift/rc_command",1)
        self.rc_unfiltered = node.create_publisher(RCMessage, "/luminosity_drone/rc_command_unfiltered",1)

        # Create publisher for publishing errors for plotting in plotjuggler 
        
        self.pid_error_pub = node.create_publisher(PIDError, "/luminosity_drone/pid_error",1)
        self.integral_pub = node.create_publisher(PIDError, "/luminosity_drone/integral",1)

    #
    def identifyCorner(self, x,y):
        if x > 0 and y > 0:
            return [-1,-1]
        elif x > 0 and y < 0:
            return [-1,1]
        elif x < 0 and y > 0:
            return [1,-1]
        return[1,1]

    def wapointGen(self, corner = [1,1]):
        x_values = np.arange(self.arena_size_x_n+7.1, self.arena_size_x, self.grid_spacing_x )[::corner[0]]
        # x_values = np.arange(arena_size_x_n, arena_size_x, grid_spacing )
        y_values = np.arange(self.arena_size_y_n, self.arena_size_y , self.grid_spacing_y)[::corner[1]]

        # Create a flag to reverse the y-values for the zigzag pattern
        i = 0
        for y in y_values:
            if i!=0:
                for ab in np.linspace(y-(4*self.corner[1]),y,6):
                    self.waypoints.append([x_values[0],ab, self.altitude])
            for x in x_values:
                self.waypoints.append([x, y, self.altitude])
            x_values = x_values[::-1]
            if i >= (len(y_values) - 1):
                if corner[1] > 0:
                    for ab in np.linspace(y,y+(1.2),3):
                        for x in x_values:
                            self.waypoints.append([x,ab, self.altitude])
                        x_values = x_values[::-1]
            i+=1

    def flip_variable(self,variable_name):
        """Flips the value of a boolean variable stored in a dictionary.

        Purpose:
        ---
        Inverts the current boolean value of a specified variable stored within a dictionary.

        Inputs:
        ---
        variable_name : str
            The name of the boolean variable to be flipped.

        Returns:
        ---
        None
            Modifies the variable's value in-place within the dictionary.
        """
        self.first_time_alien[variable_name] = not self.first_time_alien[variable_name]
        
    def identical_cluster(self, dic, coordinates):
    
        """Check no processing is done for a duplicate / same cluster"""

        should_add = True  # Flag to indicate if the cluster should be added
        for coords1 in list(dic.values()):
            
            distance = self.calculate_distance(coords1, coordinates)
            if distance < 3.5:
                
                should_add = False

                break
        return should_add# No need to compare with other clusters once a close one is found

        
    def add_cluster(self, dic, num_leds, coordinates):
            """Adds a new cluster to the dictionary, checking for duplicates first."""

            should_add = True  # Flag to indicate if the cluster should be added
            deleted_cluster = None
            for key1, coords1 in list(dic.items()):
                if key1 == num_leds:  # Skip comparing with itself
                    continue

        # Optional filtering for efficiency:
        # if key1 < num_book - 2:  # Skip if significantly fewer books
        #     continue

                distance = self.calculate_distance(coords1, coordinates)
                if distance < 3:
                    if key1 < num_leds:
                        deleted_cluster = (key1, dic[key1])
                        del dic[key1]  # Remove smaller cluster
                    else:
                        should_add = False
                    if deleted_cluster is not None:
                        self.flip_variable(deleted_cluster[0])  # Don't add the new cluster if it's smaller
                    break  # No need to compare with other clusters once a close one is found

            if should_add:
                dic[num_leds] = coordinates

    def add_cluster(self, dic, num_leds, coordinates):
            """Adds a new cluster to the dictionary, checking for duplicates first."""

            should_add = True  # Flag to indicate if the cluster should be added
            deleted_cluster = None
            for key1, coords1 in list(dic.items()):
                if key1 == num_leds:  # Skip comparing with itself
                    continue

        # Optional filtering for efficiency:
        # if key1 < num_book - 2:  # Skip if significantly fewer books
        #     continue

                distance = self.calculate_distance(coords1, coordinates)
                if distance < 3:
                    if key1 < num_leds:
                        deleted_cluster = (key1, dic[key1])
                        del dic[key1]  # Remove smaller cluster
                    else:
                        should_add = False
                    if deleted_cluster is not None:
                        self.flip_variable(deleted_cluster[0])  # Don't add the new cluster if it's smaller
                    break  # No need to compare with other clusters once a close one is found

            if should_add:
                dic[num_leds] = coordinates

            
    """Cool Function"""
    def BioLocErro(self, drone_pos, points=None):
        if points is not None:
            point = drone_pos
            shortest_distance = float('inf')
            closest_point = None
            list_cr = None

            # Calculate distance to each point in JSON data
            for name, coord in points.items():
                dist = self.calculate_distance(point[:2], coord[:2])
                if dist < shortest_distance:
                    shortest_distance = dist
                    closest_point = name
                    list_cr = coord

            # Calculate error in x and y coordinates
            error_x = abs(points[closest_point][0] - point[0])  # Absolute error to account for both directions
            error_y = abs(points[closest_point][1] - point[1])

            return error_x, error_y, list_cr
        return -999, -999, None
    def hexagon_coordinates(self, center, size):
        """Generate hexagon coordinates around a given center."""
        angles = np.linspace(0, 2*np.pi, 7)
        x = center[0] + size * np.cos(angles)
        y = center[1] + size * np.sin(angles)
        return x, y

    def generate_concentric_hexagon_pattern(self,center, num_hexagons):
        """Generate a list of concentric hexagon coordinates."""
        
        
        for _ in range(num_hexagons, 0, -1):  # Decrease the size for each concentric hexagon
            size = 0.5
            hexagon = self.hexagon_coordinates(center, size)
            self.hexagon_pattern.append(hexagon)

        return self.hexagon_pattern
    
    def whycon_poses_callback(self, msg):
        self.last_whycon_pose_received_at = self.node.get_clock().now().seconds_nanoseconds()[0]
        # print( self.last_whycon_pose_received_at )
        self.drone_whycon_pose_array = msg


    def pid_tune_throttle_callback(self, msg):
        self.Kp[2] = msg.kp * 0.01
        self.Ki[2] = msg.ki * 0.0001
        self.Kd[2] = msg.kd * 0.1

    # Similarly add callbacks for other subscribers
        
    def pid_tune_pitch_callback(self, msg):
        self.Kp[1] = msg.kp * 0.01
        self.Ki[1] = msg.ki * 0.0002
        self.Kd[1] = msg.kd * 0.1

    def pid_tune_roll_callback(self, msg):
        self.Kp[0] = msg.kp * 0.01
        self.Ki[0] = msg.ki * 0.0002
        self.Kd[0] = msg.kd * 0.1
        
        
    def calculate_distance(self,p1, p2):
        """
        Purpose:
        ---
        Calculates the Euclidean distance between two points.

        Inputs:
        ---
        p1 : tuple
            Tuple containing (x, y) coordinates of point 1
        p2 : tuple
            Tuple containing (x, y) coordinates of point 2

        Returns:
        ---
        float
            Euclidean distance between the two points
        """
        dis = 0
        for i in range(len(p1)):
            dis += (p1[i] - p2[i])**2
        return np.sqrt(dis)

# Function to cluster LEDs based on proximity
    def cluster_leds(self,centroids, max_distance_threshold):
        """
        Purpose:
        ---
        Clusters LEDs based on proximity using the centroids and a maximum distance threshold.

        Inputs:
        ---
        centroids : list
            List of centroid coordinates of detected LEDs
        max_distance_threshold : int
            Maximum distance threshold for clustering LEDs

        Returns:
        ---
        numpy array
            Array containing cluster labels for each LED
        """
        # self.node.get_logger().info(f"----------adfdsasdfsddddddddddddddddddddd")

        num_leds = len(centroids)
        labels = np.zeros(num_leds, dtype=int)
        cluster_label = 1

        for i in range(num_leds):
            if labels[i] == 0:
                labels[i] = cluster_label
                cluster_label += 1

            for j in range(i + 1, num_leds):
                if labels[j] == 0 and self.calculate_distance(centroids[i], centroids[j]) <= max_distance_threshold:
                    labels[j] = labels[i]

        return labels

    # Function to extract centroids from contours
    def get_centroids(self,contours):
        """
        Purpose:
        ---
        Extracts centroids from contours.

        Inputs:
        ---
        contours : list
            List of contours detected in the image

        Returns:
        ---
        list
            List of centroid coordinates of contours
        """
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            centroids.append((cX, cY))
        return centroids
    def led_image_alignment(self,led_centers,image):
        """Calculates alignment errors between LED centers and image center.
            Purpose:
            ---
            Determines the positional errors between the center of a set of LED centers and the center of a given image.

            Inputs:
            ---
            led_centers : list of tuples
            A list containing tuples of (x, y) coordinates representing the centers of LEDs.
            image : numpy array
            The image to be aligned with the LEDs.

            Returns:
            ---
            tuple of floats
            A tuple containing two values:
            - error_x: The horizontal alignment error between the LED center and image center.
            - error_y: The vertical alignment error between the LED center and image center.
        """
            # center_x = sum(x for x, y in led_centers) / len(led_centers)
            # center_y = sum(y for x, y in led_centers) / len(led_centers)  
        center_x=led_centers[0]
        center_y=led_centers[1]
        # Calculate the height and width of the image
        height, width = image.shape[:2]
        # cv2.imshow("image",image)

        # Calculate the center coordinates of the image
        center_x_image = width // 2
        center_y_image = height // 2
        error_x=center_x_image-center_x
        error_y=center_y_image-center_y
        return error_x,error_y
    
    
    
    def Landing_path(self,start, end, step_size):
    # Calculate the tangent of the straight line
        tangent = np.arctan2(end[1] - start[1], end[0] - start[0])

        # Calculate the Euclidean distance between start and end
        distance = np.linalg.norm(np.array(end) - np.array(start))

        # Create a path using linspace for x and y
        x_path = np.linspace(start[0], end[0], int(distance / step_size) + 1)
        y_path = np.linspace(start[1], end[1], int(distance / step_size) + 1)

        # Use the tangent to find y values
        y_path = np.tan(tangent) * (x_path - start[0]) + start[1]

        return list(list(a) for a in zip(x_path, y_path,list(range(self.z_base-5,self.z_base-4))*len(x_path)))


    def plot_square(self):
        """Generates coordinates for a square and stores them in a list.

        Purpose:
        ---
        Calculates the coordinates of points forming a square with specified dimensions and center, potentially for plotting purposes.

        Inputs:
        ---
        None (implicitly uses attributes of the class instance)

        Returns:
        ---
        None
            Modifies the `self.small_square` list in-place with the generated coordinates.
        """
    # Generate coordinates
        # x_coords = []
        # y_coords = []
        # self.small_square
        for i in range(int(-self.square_size/2), int(self.square_size/2) + 1):
            for j in range(int(-self.square_size/2), int(self.square_size/2) + 1):
                x = self.center_x + i * self.grid_length
                y = self.center_y + j * self.grid_length
                if (-self.y_max_neg <= y <= self.y_max_pos) and (-self.x_max_neg <= x <= self.x_max_pos):
                    self.small_square.append([(x), (y) , self.altitude])
                    # y_coords.append(y)
        self.small_square = self.small_square[::-1]
    def timer_callback(self, amount_delay, func=None, arg = None):
        # count = None
        if not self.elapsed_couter_start:
            self.counter = Clock().now().nanoseconds
            # self
            self.elapsed_couter_start = True

        if (Clock().now().nanoseconds - self.counter) > (amount_delay)*(1e9):
            if func is not None:
                if arg is not None:
                    try:
                        func(arg)
                    except Exception as E:
                        self.node.get_logger().error(f"Timer Callback Not Executed due to {E}")
                        pass
                elif arg is None:
                    try:
                        func()
                    except Exception as E:
                        self.node.get_logger().error(f"Timer Callback Not Executed due to {E}")
                        pass
            self.counter = Clock().now().nanoseconds
        

    def blink_beep(self, arg):
        """Controls blinking and beeping actions based on a timer and arguments.

        Purpose:
        ---
        Orchestrates a sequence of blinking and beeping actions, likely for visual and auditory signaling, using a timer and input arguments.

        Inputs:
        ---
        arg : list
            A list containing two boolean values:
            - arg[0]: A boolean flag indicating whether to continue blinking and beeping.
            - arg[1]: An integer representing the total number of blinks and beeps to perform.

        Returns:
        ---
        None
            Modifies the `arg` list and internal variables to control the blinking and beeping behavior.
        """
        if not self.count_started and arg[0]:
            self.time = Clock().now().nanoseconds
            self.count_started = True
            self.blinked = False
            
            
        if (-(self.time - Clock().now().nanoseconds) > 0.2e9) and arg[0]: 
            if self.count_aux < arg[1]:
                if not self.blinked:
             
                    self.rc_message.aux4 = 1500
                    # self.rc_message.rc_pitch = self.
                    # self.rc_pub.publish(self.rc_message)
                    self.rc_message.aux3 = 2000
                # self.rc_pub.publish(self.rc_message)
                    self.blinked = True
                    self.node.get_logger().error(f"{self.rc_message.aux3}, {self.rc_message.aux4}")
            if (-(self.time - Clock().now().nanoseconds) > 0.4e9) and arg[0]:
                if self.count_aux < arg[1]:
                    self.node.get_logger().info(f"Bllinking - {(Clock().now().nanoseconds - self.time)/1e9}")
                    
                    self.rc_message.aux4 = 2000
                    # self.rc_pub.publish(self.rc_message)
                    self.rc_message.aux3 = 1000
                    # self.rc_pub.publish(self.rc_message)
                    self.node.get_logger().error(f"-----------{self.rc_message.aux3}, {self.rc_message.aux4}")
                    self.count_aux += 1
                    self.blinked = False
                    
                else:
                    arg[0] = False
                    self.count_started = False
                    self.count_aux = 0

                    

                self.time = Clock().now().nanoseconds


        

    
    def image_callback(self, msg):
        """Processes incoming images to detect LEDs, navigate the drone, and trigger actions.

    Purpose:
    ---
    1. Processes images to detect LEDs, potentially representing alien organisms.
    2. Aligns the drone with detected LEDs and updates its position.
    3. Generates waypoints for further exploration.
    4. Triggers actions like blinking, beeping, and logging coordinates based on LED alignment and time.
    5. Publishes messages containing organism type and position.

    Inputs:
    ---
    (No direct inputs for the callback function. It's triggered by incoming image data.)

    Returns:
    ---
    None
        Modifies global variables and publishes messages to control drone behavior and communication.

    """
        # if self.callback_count < self.callback_frequency:
        # # self.node.get_logger().info(f"type image{type(msg)}")
        #     self.callback_count += 1
        if Clock().now().nanoseconds - self.total_time > 5e9 and self.detect_Led:
            image_msg = self.bridge.imgmsg_to_cv2(msg)
            cv2.imshow("data",image_msg)
            cv2.waitKey(1)
            image=image_msg
            # self.node.get_logger().info("run")
            if image is None:
                print("Error: Unable to load the image.")
                return
            # cv2.imshow("image",image)
            # self.node.get_logger().info(f"-----------------{image_msg.shape()}")
            cv2.imwrite("image.jpg",image)
            drone_position=[0,0,0]
            cv2.waitKey(1)
            if self.drone_whycon_pose_array.poses:
                x=self.drone_whycon_pose_array.poses[0].position.x
                y=self.drone_whycon_pose_array.poses[0].position.y
                z=self.drone_whycon_pose_array.poses[0].position.z
                drone_position=[x,y,z]

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # self.node.get_logger().info(f"-------------------------")

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)

            # Threshold the blurred image to detect bright regions
            threshold_value = 200
            thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)[1]

            # Perform erosion and dilation to remove small noise
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=1)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Minimum distance to consider contours as separate LEDs
            min_distance = 20

            # Filter contours that are too close
            separated_contours = []
            for i, c1 in enumerate(contours):
                for j, c2 in enumerate(contours):
                    if i != j and cv2.norm(cv2.minEnclosingCircle(c1)[0], cv2.minEnclosingCircle(c2)[0]) > min_distance:
                        separated_contours.append(c1)

            # Draw contours of separated LEDs
            for contour in separated_contours:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            

            # Display the image with separated contours
            # print(len(contours))

            centroids = self.get_centroids(contours)

            labels = self.cluster_leds(centroids, max_distance_threshold=350)


            clustered_centroids = {}
            for i, label in enumerate(labels):
                if label not in clustered_centroids:
                    clustered_centroids[label] = [centroids[i]]
                else:
                    clustered_centroids[label].append(centroids[i])

            organism_types = {
                # 1: "alien_a",
                2: "alien_a",
                3: "alien_b",
                4: "alien_c",
                5: "alien_d"
            }
            dict_aliens={}
            # centroid_data = {}
            num_leds  = 0
            # big_led = None
            for cluster_label, cluster_centroids in clustered_centroids.items():

                num_leds = len(cluster_centroids)
                organism_type = organism_types.get(num_leds, f"Unknown (LEDs: {num_leds})")

                centroid = np.mean(cluster_centroids, axis=0, dtype=int)
                # centroid_str = f"({centroid[0]:.4f}, {centroid[1]:.4f})"  # Format centroid coordinates

                # Populate the dictionary with centroid information
                dict_aliens = {
                    'Organism Type': organism_type,
                    'Centroid': centroid.tolist()  # Convert centroid to a list for better storage
                }

            # Print or use the centroid_data dictionary as needed
            # print(dict_aliens)
            for alien,centr in dict_aliens.items():
                organism_type=alien
                centroid=centr
            
                
            if self.waypoint_update and len(self.coor_alien)>0:
                if (self.calculate_distance(drone_position, self.coor_alien.get(list(self.coor_alien.keys())[-1], [900,900,900])) > 2.5):
                    self.alignment_needed = True


            if self.droneInRange():
                align = False
            # self.node.get_logger().info("asdfghjqwertyuzxcvb")
                if num_leds>0:
                    # self.loop_start_time += Clock().now().nanoseconds
                    
                    # c = 0
                    self.publish_values = False
                    self.node.get_logger().error(f"This frame has {num_leds} leds")
                    if len(self.coor_alien)>0:
                        
                        if self.identical_cluster(self.coor_alien, drone_position) and self.total_leds>len(self.coor_alien):
                    # self.alignment_needed = True
                            self.waypoint_update = False
                            self.node.get_logger().error(f"wapoint not further {num_leds}")
                        # else:
                            # self.waypoint_update = True
                        
                    else:
                        self.waypoint_update = False
                    """----------------"""
                    self.LED_mili=drone_position 
                    """----------------"""

                    if not self.believe and ((self.hexagon_inde <= len(self.hexagon_waypoints)-1) or (self.hexagon_inde == 0 and self.hexagon_waypoints == [])):#stop generating new sqaure pattern if centre i aligned
                        # num_concentric_hexagons = 1
                        
                        self.start_point = drone_position

                        # Generate concentric hexagon pattern coordinates
                        self.center_x = drone_position[0]
                        self.center_y = drone_position[1]

                        self.plot_square()
                        
                        self.hexagon_inde = 0
                        
                        self.hexagon_waypoints = self.small_square
                    self.node.get_logger().warn(f"Drone Position of frame -{drone_position}")
                    # print(drone_position," - Current Drone Position")
                    # self.node.get_logger().error("till here")
                    # self.node.get_logger().info(f"---1______{centroid}")
                    x_error,y_error=self.led_image_alignment(centroid,thresh)
                    # print("error",x_error,y_error)
                    # print("old set_points",self.set_points)
                    # self.node.get_logger().info("---2______")
                    detected_leds = []
                    
                    """Cool Logic but Chances"""
                    # err_x, err_y, coor = self.BioLocErro(drone_position, points)
                    # if (err_x <= 0.8)and err_y <= 0.8 and self.current_led is not None:
                    #     if num_leds in (self.image_errors.keys()) and self.shm and self.current_led>= num_leds:
                    #         self.image_errors[self.current_led] = 150
                    #         if coor is not None and self.current_led in self.coor_alien.keys():
                    #             # prev = self.coor_alien[self.current_led]
                    #             # self.coor_alien[self.current_led] = coor
                    #             self.set_points[0], self.set_points[1] = coor[0], coor[1]
                    #             self.waypoint_tolerance = 0.25
                    #         self.ek_aur_var = True
                    #         self.shm = True
                    #         self.node.get_logger().error(f"Current - Led{self.current_led}")
                    if (abs(x_error)<self.image_errors_x.get(num_leds, 150) and abs(y_error) < self.image_errors_y.get(num_leds, 150)) :
                        if self.shm:
                            self.believe=True
                            if self.ek_aur_var:
                                self.loop_start_time = Clock().now().nanoseconds 
                                c = 0
                                self.ek_aur_var = False
                            self.node.get_logger().info(f"Time Passed - {(Clock().now().nanoseconds - self.loop_start_time)/1e9}")
                            self.node.get_logger().info(f"---{self.waypoint_update}")
                            if Clock().now().nanoseconds - self.loop_start_time >= 0.5e9 and self.believe and not self.blink_and_beep[0] and not self.waypoint_update:
                                if len(self.coor_alien.keys()) > 0:
                                    
                                    big_led = list(self.coor_alien.keys())[-1]

                                    if len(self.coor_alien)==self.total_leds:
                                        self.waypoints=self.Landing_path(drone_position,self.landing_cooridnates,1)
                                        self.node.get_logger().error(f"new waypoints yeeee-------------{self.waypoints}")
                                        self.current_waypoint_index = 0
                                    
                                    
                                    self.node.get_logger().error(f"{self.coor_alien[big_led][0]+self.whycon_bias_x},{self.coor_alien[big_led][1]+self.whycon_bias_y}, {self.coor_alien[big_led][2]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.node.get_logger().error(f"{self.coor_alien[big_led]} , | {self.BioLocErro(self.coor_alien[big_led], points)}")
                                    self.waypoint_update = True
                                    self.believe = False
                                    self.shm = False
                                    biolocation_message = Biolocation()
                                    if big_led not in self.published_leds:
                                        self.blink_and_beep = [True, big_led]
                                        if self.coor_alien[big_led][0] <0:
                                            self.whycon_bias_y /= 2
                                        biolocation_message.organism_type = organism_types.get(big_led)
                                        biolocation_message.whycon_x = (self.coor_alien[big_led][0]+self.whycon_bias_x) #bias of focal lenght and cemra tilt for ceiling
                                        biolocation_message.whycon_y = (self.coor_alien[big_led][1]+self.whycon_bias_y) #bias of focal lenght and cemra tilt for ceiling
                                        biolocation_message.whycon_z = self.coor_alien[big_led][2]
                                        self.biolocation_publisher.publish(biolocation_message)
                                        self.published_leds.append(big_led)
                                        if self.coor_alien[big_led][0] <0:
                                            self.whycon_bias_y *= 2
                                        # self.image_errors = { 2: 90, 3: 90, 4: 90, 5: 90}
                                    # self.coor_alien[self.current_led] = prev



                                    
                                    cv2.imwrite("led_detected.jpg", thresh)
                                    cv2.imwrite("led_detected_everytime.jpg", image) 

                                    self.centroid_of_LEDs=drone_position
                                    self.alignment_needed = False

                            if self.first_time_alien.get(num_leds, False):

                                self.add_cluster(self.coor_alien, num_leds, drone_position)

                                self.first_time_alien[num_leds] = False
                                
                                self.node.get_logger().error(f"{num_leds} - found alien found alien - {drone_position}")
                                self.node.get_logger().error(f"{num_leds} - found alien found alien - {drone_position}")
                                self.node.get_logger().error(f"{num_leds} - found alien found alien - {drone_position}")
                                self.node.get_logger().error(f"{num_leds} - found alien found alien - {drone_position}")
                                self.node.get_logger().error(f"{num_leds} - found alien found alien - {drone_position}")
                                self.node.get_logger().error(f"{num_leds} - found alien found alien - {drone_position}")
                                self.node.get_logger().error(f"{num_leds} - found alien found alien - {drone_position}")

                            "----------------"
                            # if len(self.coor_alien.keys()) > 0:
                            #     big_led = max(self.coor_alien.keys())
                            
                            # if len(self.coor_alien) >= 1 and big_led in detected_leds:
                            #     big_led = max(self.coor_alien.keys())
                            #     detected_leds.append(big_led)
                            "----------------"
                            
                    elif self.alignment_needed:

                        if self.identical_cluster(self.coor_alien, drone_position):
                            if abs(x_error)>self.image_errors_x.get(num_leds, 150):
                                self.set_points[0]=drone_position[0]-self.correction_scale_x*x_error
                                # self.waypoint_tolerance = 0.2
                                self.tolerance_list = [0.2,0.2,0.5]
                                self.node.get_logger().error("Trying to Correct Error_x ")
                            if abs(y_error)>self.image_errors_y.get(num_leds, 150):
                                self.set_points[1]=drone_position[1]-self.correction_scale_y*y_error
                                # self.waypoint_tolerance = 0.2
                                self.tolerance_list = [0.2,0.2,0.5]
                                self.node.get_logger().error("Trying to Correct Errro_y")
                            self.ek_aur_var = True
                            self.shm = True
                            # self.Ki = [ 170.00*0.0002  , 140*0.0002 , 0.0208 ]
                        # if self.current_led is not None:
                        #     if num_leds >= self.current_led and num_leds in self.image_errors.keys():
                        #         self.current_led = num_leds
                        # elif num_leds in self.image_errors.keys():
                        #     self.current_led = num_leds
                            
                        
                else: 

                    self.believe = False
                    self.publish_values = True
        # else: 
        #     time_elapsed = Clock().now().nanoseconds - self.callback_timer
        #     print("image callback = ",self.callback_count, time_elapsed/1e9)
        #     if (time_elapsed ) >= self.callback_window:
        #         # print("----------------------", )
        #         self.callback_count = 0
        #         self.callback_timer = Clock().now().nanoseconds
        #     else:
        #         print("+1")
        #         pass
        
             
                    
    def droneInRange(self):
        """Determines whether the drone is within acceptable range of a waypoint.

            Purpose:
            ---
            Checks if the drone's current position is within a specified self.tolerance_list of a target waypoint, likely for navigation accuracy.

            Inputs:
            ---
            (No direct inputs. The function operates on internal state variables.)

            Returns:
            ---
            bool
                True if the drone is within range of the waypoint, False otherwise.
        """
        if self.tolerance_list is None:
            
            if all(abs(error) < self.waypoint_tolerance for error in self.error):
                return True
            return False
        else:
            if (abs(self.error[0]) < self.tolerance_list[0]) and (abs(self.error[1]) < self.tolerance_list[1]) and (abs(self.error[2] < self.tolerance_list[2])):
                # self.tolerance_list = None
                return True
            # self.tolerance_list = None
            return False
    

    def pid(self):          # PID algorithm

        # 0 : calculating Error, Derivative, Integral for Roll error : x axis
        try:
            self.error[0] = (self.drone_whycon_pose_array.poses[0].position.x - self.set_points[0])
            
        # Similarly calculate error for y and z axes 
            self.error[1] = (self.drone_whycon_pose_array.poses[0].position.y - self.set_points[1])
            self.error[2] = self.drone_whycon_pose_array.poses[0].position.z - self.set_points[2] + 0.31
            x = self.drone_whycon_pose_array.poses[0].position.x
            y = self.drone_whycon_pose_array.poses[0].position.y 
            z = self.drone_whycon_pose_array.poses[0].position.z
        
        except:
            pass
        
        
        # Calculate derivative and intergral errors. Apply anti windup on integral error (You can use your own method for anti windup, an example is shown here)
        self.derivative[0] = self.error[0] - self.prev_error[0]
        self.integral[0] = (self.integral[0] + self.error[0])
        if self.integral[0] > SUM_ERROR_ROLL_LIMIT:
            self.integral[0] = SUM_ERROR_ROLL_LIMIT
        if self.integral[0] < -SUM_ERROR_ROLL_LIMIT:
            self.integral[0] = -SUM_ERROR_ROLL_LIMIT

        # Save current error in previous error
        self.prev_error[0] = self.error[0]


        # 1 : calculating Error, Derivative, Integral for Pitch error : y axis

        self.derivative[1] = self.error[1] - self.prev_error[1]
        self.integral[1] = (self.integral[1] + self.error[1])
        if self.integral[1] > SUM_ERROR_PITCH_LIMIT:
            self.integral[1] = SUM_ERROR_PITCH_LIMIT
        if self.integral[1] < -SUM_ERROR_PITCH_LIMIT:
            self.integral[1] = -SUM_ERROR_PITCH_LIMIT

        self.prev_error[1] = self.error[1]



        # 2 : calculating Error, Derivative, Integral for Alt error : z axis
        self.derivative[2] = self.error[2] - self.prev_error[2]
        self.integral[2] = (self.integral[2] + self.error[2])
        if self.integral[2] > SUM_ERROR_THROTTLE_LIMIT:
            self.integral[2] = SUM_ERROR_THROTTLE_LIMIT
        if self.integral[2] < -SUM_ERROR_THROTTLE_LIMIT:
            self.integral[2] = -SUM_ERROR_THROTTLE_LIMIT        
        self.prev_error[2] = self.error[2]

        # Write the PID equations and calculate the self.rc_message.rc_throttle, self.rc_message.rc_roll, self.rc_message.rc_pitch

        self.out_throttle = int(BASE_THROTTLE + self.error[2]*self.Kp[2] + self.derivative[2] * self.Kd[2] + self.integral[2]*self.Ki[2])
        self.out_pitch = int(BASE_PITCH + (self.error[1]*self.Kp[1] + self.derivative[1] * self.Kd[1] + self.integral[1]*self.Ki[1]))
        self.out_roll = int(BASE_ROLL - ((self.error[0]*self.Kp[0]) + self.derivative[0] * self.Kd[0] + self.integral[0]*self.Ki[0]))
        
        if Clock().now().nanoseconds - self.total_time > 100e9 and self.detect_Led :
            self.detect_Led = False
            self.waypoints = self.Landing_path([x,y,z],self.landing_cooridnates,1)
            # self.waypoint_update = True
            self.set_points = self.waypoints[0]
            self.current_waypoint_index = 0
            self.waypoint_update = True
            # self.detect_Led = False

        self.publish_data_to_rpi( roll = self.out_roll, pitch = self.out_pitch, throttle = self.out_throttle)

        #Replace the roll pitch and throttle values as calculated by PID 
        
        
        # Publish alt error, roll error, pitch error for plotjuggler debugging

        self.pid_error_pub.publish(
            PIDError(
                roll_error=float(self.error[0]),
                pitch_error=float(self.error[1]),
                throttle_error=float(self.error[2]),
                yaw_error=-0.5,
                zero_error=0.0,
            )
        )

        self.integral_pub.publish(
            PIDError(
                roll_error=float(self.integral[0]),
                pitch_error=float(self.integral[1]),
                throttle_error=float(self.integral[2]),
                yaw_error=0.5,
                zero_error=float(1500),
            )
        )

        self.rc_unfiltered.publish(RCMessage(
            rc_throttle = int(self.out_throttle),
            rc_roll = int(self.out_roll),
            rc_pitch = int(self.out_pitch),
            rc_yaw = int(1500)

        )
    )


    def publish_data_to_rpi(self, roll, pitch, throttle):

        self.rc_message.rc_throttle = int(throttle)
        self.rc_message.rc_roll = int(roll)
        self.rc_message.rc_pitch = int(pitch)

        # Send constant 1500 to rc_message.rc_yaw
        self.rc_message.rc_yaw = int(1500)

        """
        import scipy.signal

# Your existing code with defined variables

span = 15
order = 4  # Increasing the filter order for a steeper roll-off
fs = 60
fc = 2  # Lowering the cutoff frequency to preserve more of the waypoints signal

# Loop through each signal (roll, pitch, throttle)
for index, val in enumerate([roll, pitch, throttle]):
    DRONE_WHYCON_POSE[index].append(val)
    
    if len(DRONE_WHYCON_POSE[index]) == span:
        DRONE_WHYCON_POSE[index].pop(0)
        
    if len(DRONE_WHYCON_POSE[index]) == span - 1:
        # Butterworth filter parameters
        nyq = 0.5 * fs
        wc = fc / nyq
        b, a = scipy.signal.butter(N=order, Wn=wc, btype='lowpass', analog=False, output='ba')

        # Apply the filter
        filtered_signal = scipy.signal.lfilter(b, a, DRONE_WHYCON_POSE[index])

        # Modify rc_message based on the filtered signal
        if index == 0:
            self.rc_message.rc_roll = int(filtered_signal[-1])
        elif index == 1:
            self.rc_message.rc_pitch = int(filtered_signal[-1])
        elif index == 2:
            self.rc_message.rc_throttle = int(filtered_signal[-1])

        """

        # BUTTERWORTH FILTER
        span = 15
        for index, val in enumerate([roll, pitch, throttle]):
            DRONE_WHYCON_POSE[index].append(val)
            if len(DRONE_WHYCON_POSE[index]) == span:
                DRONE_WHYCON_POSE[index].pop(0)
            if len(DRONE_WHYCON_POSE[index]) != span-1:
                return
            order = 2
            fs = 60
            fc = 5
            nyq = 0.5 * fs
            wc = fc / nyq
            b, a = scipy.signal.butter(N=order, Wn=wc, btype='lowpass', analog=False, output='ba')
            filtered_signal = scipy.signal.lfilter(b, a, DRONE_WHYCON_POSE[index])
            if index == 0:
                self.rc_message.rc_roll = int(filtered_signal[-1])
            elif index == 1:
                self.rc_message.rc_pitch = int(filtered_signal[-1])
            elif index == 2:
                self.rc_message.rc_throttle = int(filtered_signal[-1])


        if self.rc_message.rc_roll >= MAX_ROLL:     #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_roll = MAX_ROLL
        elif self.rc_message.rc_roll <= MIN_ROLL:
            self.rc_message.rc_roll = MIN_ROLL

        if self.rc_message.rc_pitch >= MAX_PITCH:     #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_pitch = MAX_PITCH
        elif self.rc_message.rc_pitch <= MIN_PITCH:
            self.rc_message.rc_pitch = MIN_PITCH

        if self.rc_message.rc_throttle >= MAX_THROTTLE:     #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_throttle = MAX_THROTTLE
        elif self.rc_message.rc_throttle <= MIN_THROTTLE:
            self.rc_message.rc_throttle = MIN_THROTTLE
        # Similarly add bounds for pitch yaw and throttle 
        if self.disarm and Clock().now().nanoseconds-self.total_time < 2e9:
            self.rc_message.aux4 = 1500 
            self.rc_message.aux3 = 2000
        else:
            self.rc_message.aux4 = 2000
            self.rc_message.aux3 = 1000
            
        self.blink_beep(self.blink_and_beep)
        self.rc_pub.publish(self.rc_message)
        # if self.total_time > 50e9:
        #     self.Ki = [ 150.00*0.0002  , 130*0.0002 , 0.0208 ]
        # if ((self.total_time/1e9) // 1) % 2 == 0:
        
        # self.timer_callback(amount_delay=1, func=self.node.get_logger().warn,arg=f"{self.rc_message.rc_roll}, {self.rc_message.rc_pitch}, {self.rc_message.rc_throttle}, {self.integral[0]}, {self.integral[1]}")
        # self.node.get_logger().warn(f"{self.rc_message.rc_roll}, {self.rc_message.rc_pitch}, {self.rc_message.rc_throttle}, {self.integral[0]}, {self.integral[1]}")
        # self.node.get_logger().warn(f"{self.derivative[1]}, {self.derivative[1]*self.Kd[1]}, {self.derivative[1] * self.Kd[1] + self.integral[1]*self.Ki[1]}")

       
    

    # This function will be called as soon as this rosnode is terminated. So we disarm the drone as soon as we press CTRL + C. 
    # If anything goes wrong with the drone, immediately press CTRL + C so that the drone disamrs and motors stop 

    def shutdown_hook(self):
        self.node.get_logger().info("Calling shutdown hook")
        self.disarmfu()

    # Function to arm the drone 

    def arm(self):
        self.node.get_logger().info("Calling arm service")
        self.commandbool.value = True
        self.future = self.arming_service_client.call_async(self.commandbool)

    # Function to disarm the drone 

    def disarmfu(self):
        try:
            self.node.get_logger().info("Calling disarm service")
            landing_duration = 5  # Duration for soft landing in seconds
            landing_steps =  10# Number of steps to reduce throttle gradually

            throttle_reduction = 12
            current_throttle = self.rc_message.rc_throttle

            for _ in range(landing_steps):
                
                current_throttle -= throttle_reduction

                self.rc_message.rc_throttle = int(current_throttle)
                self.rc_pub.publish(self.rc_message)
                time.sleep(landing_duration / landing_steps)

            # Ensure the throttle is set to minimum after landing
            self.rc_message.rc_throttle = MIN_THROTTLE
            self.rc_pub.publish(self.rc_message)
        except Exception:
            # self.commandbool.value = False
            # self.future = self.arming_service_client.call_async(self.commandbool)
            # self.node.get_logger().info("Calling disarm service\n"*10)
            self.rc_message.rc_throttle = MIN_THROTTLE
            self.rc_pub.publish(self.rc_message)
        finally:
            self.commandbool.value = False
            self.future = self.arming_service_client.call_async(self.commandbool)
            self.node.get_logger().info("Calling disarm service\n"*10)
            # Create the disarm function

    def land(self):
        try:
            self.rc_message.rc_throttle = self.rc_message.rc_throttle - 10
            time.sleep(0.2)
            self.rc_message.rc_throttle = MIN_THROTTLE
            self.rc_pub.publish(self.rc_message)
        except Exception:
            pass
        finally:
            self.commandbool.value = False
            self.future = self.arming_service_client.call_async(self.commandbool)
            self.node.get_logger().info("\nCalling landing Service"*10)

    # def executeHybridLand(self, x, y, z, landing_waypoints, landing_waypoint_index, disarm, landed):
    
    #         if self.total_leds == len(self.coor_alien):
    #             if len(landing_waypoints) == 0:
    #                 z_step = 1  # Adjust the decrement per step as needed

    #                 for i in np.arange(z, self.z_base, z_step):
    #                     landing_waypoints.append([-10.3, -10.3, i])
    #             if landing_waypoint_index < len(landing_waypoints):
    #                 self.set_points = landing_waypoints[landing_waypoint_index]
    #                 landing_waypoint_index += 1
    #             elif landing_waypoint_index >= len(landing_waypoints):
    #                 disarm = True
    #         else:
    #             if len(landing_waypoints) == 0:
    #                 landing_path = self.Landing_path([x, y, z], [-10.3, -10.3, 23], 1)

    #                 z_step = 1  # Adjust the decrement per step as needed

    #                 for i in np.arange(z, self.z_base, z_step):
    #                     landing_waypoints.append([-10, -10, i])
    #                 landing_path += landing_waypoints
    #             if landing_waypoint_index < len(landing_path):
    #                 self.set_points = landing_path[landing_waypoint_index]
    #                 landing_waypoint_index += 1
    #             elif landing_waypoint_index >= len(landing_path):
    #                 disarm = True

    #             elif disarm and self.droneInRange():
    #                 if not landed:
    #                     self.land()
    #                     landed = True
    #                     self.node.get_logger().info(f"Drone Landed, Current Position = [{x},{y},{z}]")
    def handle_landing(self, x, y, z):
        if self.land and not self.disarm and self.droneInRange():
            if self.total_leds == len(self.coor_alien):
                if len(self.landing_waypoints) == 0:
                    z_step = 1.3  # Adjust the decrement per step as needed

                    for i in np.arange(z, self.z_base+2, z_step):
                        self.landing_waypoints.append([self.landing_cooridnates[0], self.landing_cooridnates[1], i])
                if self.landing_waypoint_index < len(self.landing_waypoints):
                    self.set_points = self.landing_waypoints[self.landing_waypoint_index]
                    self.landing_waypoint_index += 1
                elif self.landing_waypoint_index >= len(self.landing_waypoints):
                    self.disarm = True
            else:
                if len(self.landing_waypoints) == 0:
                    self.landing_path = self.Landing_path([x, y, z], self.landing_cooridnates, 1)

                    z_step = 1.3  # Adjust the decrement per step as needed

                    for i in np.arange(z, self.z_base+2, z_step):
                        self.landing_waypoints.append([self.landing_cooridnates[0], self.landing_cooridnates[1], i])
                    self.landing_path += self.landing_waypoints
                if self.landing_waypoint_index < len(self.landing_path):
                    self.set_points = self.landing_path[self.landing_waypoint_index]
                    self.landing_waypoint_index += 1
                elif self.landing_waypoint_index >= len(self.landing_path):
                    self.disarm = True

        elif self.disarm and self.droneInRange():
            if not self.landed:
                self.land()
                self.landed = True
                self.node.get_logger().info(f"Drone Landed, Current Position = [{x},{y},{z}]")



        


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('controller')
    node.get_logger().info(f"Node Started")
    node.get_logger().info("Entering PID controller loop")

    controller = DroneController(node)
    node.get_logger().info(f"{controller.last_whycon_pose_received_at}")
    time.sleep(0.5)
    controller.arm()
    node.get_logger().info("Armed")
    time.sleep(0.2)

    clock = Clock()

    loop_count = 0
    # start_time = clock.now().nanoseconds
    loop_rate = 30 #freuqncy
    loop_duration = 1.0 / loop_rate
    land = False

    # disarm = False
    # global data
    # controller.ledBlinkBuzzerBeep(3)
    sec = 0
    initalised = False
    
    try:
        start_time_inside_condition = clock.now().nanoseconds
        while rclpy.ok():
            loop_start_time = clock.now().nanoseconds
            if controller.drone_whycon_pose_array.poses:
                x = controller.drone_whycon_pose_array.poses[0].position.x
                y = controller.drone_whycon_pose_array.poses[0].position.y
                z = controller.drone_whycon_pose_array.poses[0].position.z
                if not initalised:
                    corner = controller.identifyCorner(x,y)
                    controller.corner = corner
                    controller.wapointGen(corner)
                    
                    controller.set_points = [[controller.waypoints[0][0], controller.waypoints[0][1], controller.z_base-5]]
                    # if corner[0] < 0:
                    #     if abs(corner[1])>0:
                    #         controller.set_points = [x-corner[0], y+(corner[1]), controller.z_base-4]
                    #         # controller.waypoints.insert(0,[controller.waypoints[0][0], controller.waypoints[0][1]+(corner[1]), controller.z_base-3] )
                    #     # elif corner[1]<0:
                    #     #     controller.set_points = [x-corner[0]*2, y+(corner[1]*2), controller.z_base-3]
                    #     #     controller.waypoints.insert(0,[controller.waypoints[0][0], controller.waypoints[0][1]+(corner[1]*1.5), controller.z_base-3] )
                    # elif corner[0] > 0:
                    #     if abs(corner[1])>0:
                    #         # controller.set_points = [x+corner[0], y+(corner[1]), controller.z_base-4]
                    #         controller.set_points = [[controller.waypoints[0][0], controller.waypoints[0][1]+(corner[1]), controller.z_base-4]]
                    #     # elif corner[1]<0:
                    #     #     controller.set_points = [x+corner[0]*2, y+(corner[1]*2), controller.z_base-3]
                    #     #     controller.waypoints.insert(0,[controller.waypoints[0][0], controller.waypoints[0][1]+(corner[1]*1.5), controller.z_base-3] )


                    node.get_logger().info("\nGenerated Waypoints"*2)

                    # node.get_logger().info(f"\nGenerated Waypoints = {controller.waypoints}")
                    node.get_logger().info(f"\nGenerated Setpoint = {controller.set_points}")
                    initalised = True
        
            controller.pid()
            # node.get_logger().info("Yepps-2")
            
            if controller.drone_whycon_pose_array.poses:
                x = controller.drone_whycon_pose_array.poses[0].position.x
                y = controller.drone_whycon_pose_array.poses[0].position.y
                z = controller.drone_whycon_pose_array.poses[0].position.z


                if controller.droneInRange():

                    if land and not controller.disarm and controller.droneInRange():
                        
                        controller.handle_landing(x, y, z)
                    #     if controller.total_leds == len(controller.coor_alien):
                    #         if len(landing_waypoints) == 0:
                    #             z_step = 1  # Adjust the decrement per step as needed

                    #             for i in np.arange(z, controller.z_base, z_step):
                    #                 landing_waypoints.append([-10, -10, i])
                    #         if landing_waypoint_index < len(landing_waypoints):
                    #             controller.set_points = landing_waypoints[landing_waypoint_index]
                    #             landing_waypoint_index += 1
                    #         elif landing_waypoint_index >= len(landing_waypoints):
                    #             disarm = True
                    #     else:
                    #         if len(landing_waypoints) == 0:
                    #             landing_path = controller.Landing_path([x, y, z], [-10.3,-10.3,23],1)

                    #             z_step = 1  # Adjust the decrement per step as needed

                    #             for i in np.arange(z, controller.z_base, z_step):
                    #                 landing_waypoints.append([-10, -10, i])
                    #             landing_path += landing_waypoints
                    #         if landing_waypoint_index < len(landing_path):
                    #             controller.set_points = landing_path[landing_waypoint_index]
                    #             landing_waypoint_index += 1
                    #         elif landing_waypoint_index >= len(landing_path):
                    #             disarm = True

                    elif controller.disarm and controller.droneInRange():
                        # node.get_logger().info(f"LAnding On{x},{y},{z}")
                        if not controller.landed:
                            controller.land()
                            controller.landed = True
                            node.get_logger().info(f"Drone Landed , Current Position = [{x},{y},{z}")
                        

                        
                  # Reset the start time
                    node.get_logger().info(f"Reached {controller.set_points}, Current Index = {controller.current_waypoint_index} ,Current Position =  {x},{y},{z}, Error Tolerance = {controller.waypoint_tolerance}")

                    controller.waypoint_tolerance = 0.5
                    controller.tolerance_list = [0.7,0.6,1]
                    
                    if (not land) and controller.current_waypoint_index< len(controller.waypoints) and controller.waypoint_update and not controller.blink_and_beep[0]:
                        controller.set_points = controller.waypoints[controller.current_waypoint_index]
                        controller.current_waypoint_index += 1

                    elif not land and controller.droneInRange() and controller.waypoint_update and not controller.blink_and_beep[0]:
                        #for initiating PID based land
                        land = True

                    elif not controller.waypoint_update and ((controller.hexagon_inde) < len(controller.hexagon_waypoints)) and not controller.believe and controller.publish_values:
                        node.get_logger().info(f"Trying to Stabilize around LED")
                        controller.waypoint_tolerance = 0.4
                        controller.tolerance_list = [0.3,0.3,0.8]
                        if controller.hexagon_inde == 0:
                            node.get_logger().info(f"Small Area Square Coordinates{controller.hexagon_waypoints}")
                        controller.set_points = controller.hexagon_waypoints[controller.hexagon_inde]
                        controller.hexagon_inde +=1
                        
 
            # if node.get_clock().now().to_msg().sec - controller.last_whycon_pose_received_at > 1:
            if node.get_clock().now().to_msg().sec - controller.last_whycon_pose_received_at > 1:
                node.get_logger().error("Unable to detect WHYCON poses")

                # Increment loop count
            loop_count += 1

            # Calculate time elapsed
            elapsed_time = clock.now().nanoseconds - loop_start_time
            remaining_time = loop_duration - (elapsed_time / 1e9)  # Convert nanoseconds to seconds

            # Check if one second has elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
                pass
                
            # Check if one second has elapsed
            if loop_count >= loop_rate:
                # node.get_logger().error(f"Loop count in 1 second: {loop_count}")
                sec+=1
                loop_count = 0

            rclpy.spin_once(node) # Sleep for 1/30 secs
        

    except Exception as err:
        # node.get_logger().error("Unable to detect WHYCON poses")
        raise(err)
        # node.get_logger().info(f"{err}")
        # raise(err)

    finally:
        node.get_logger().info("Fianly:Clause")
        node.get_logger().info(f"Total Time of Execution = {(Clock().now().nanoseconds - controller.total_time)/1e9}")

        controller.shutdown_hook()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



