import argparse
import logging
import time
import math
import multiprocessing
import asyncio
import cv2
import zmq
import zmq.asyncio as aiozmq
import json
import queue
from utils import Pose, Point, translate
from utils import PIDController
from collections import deque
from enum import Enum
from thymiodirect import Connection, Thymio

class Modes(Enum):
    VISUAL_SERVORING = 1
    STOPPED = 2
    RETURN = 3
    MOVING = 4
    AVOIDING = 5
    EMERGENCY_STOP = 6

class ThymioRobot:
    """
    This class represents the instance of a Thymio Robot. This class implements methods to help control the robot,
    handles with data coming from the serial connection, implements odometry calculations and implements the logic
    required for this project.

    To use this robot program:
    - run_loop(): This method set up the asyncio event loop and starts core tasks to the logic.
    """
    def __init__(self, L: float, *, refreshing_rate: float = 0.01, refreshing_coverage: dict | list = None, debug: bool = False, remote_vision: bool = True, remote_addr: str = "127.0.0.1:5555"):
        """
        Initializes and parametrizes the Thymio robot.

        :param L: Distance between the middle of the wheel in millimeters.
        :param refreshing_rate: Default is 0.01. Refreshing rate of robot variables in seconds.
        :param refreshing_coverage: Default is None. List of variables to refresh. If none every variable is refreshed.
        :param debug: Default is False. If true display extra high rate debug messages. Only for debug purposes.
        :param remote_vision: Default is True. Enables the robot to wait for the processed from a remote vision processor. If false does it locally.
        :param remote_addr: Default is "192.168.1.107:5555". Address to the remove vision processor publisher/server.
        """
        # Setting up logger for the ThymioRobot
        self.logger = logging.getLogger('Thymio Robot')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Setup Thymio connection
        port = Connection.serial_default_port()
        self.robot = Thymio(serial_port=port,
                    refreshing_rate=refreshing_rate,
                    refreshing_coverage=refreshing_coverage,
                    on_connect=lambda node_id: self.logger.info(f"ThymioRobot connected successfully on node -> {node_id}"),
                    on_comm_error=lambda error: self.logger.error(f"Failed to connect to Thymio! Error Msg: {error}"))
        self.robot.connect()
        self.node = self.robot.first_node()

        # Setup refresh handler function (is executed everytime thymiodirect refreshes robot values)
        self.prev_time = time.time_ns()
        self.robot.set_variable_observer(self.node, self._refresh_handler)

        # Odometry variables
        self.pose = Pose(0, 0, 0)
        self.L = L

        # Location/Path variables
        self.pos_threshold = 2
        self.angle_threshold_lower = math.pi / 180
        self.angle_threshold_upper = math.pi / 45

        # Buffer of last values of speed (for 10ms of resampling this retains 10s of data)
        self.last_speeds = deque(maxlen=1000)

        # State control
        self.current_mode = Modes.STOPPED
        self.remote_addr = remote_addr
        self.remote_vision = remote_vision

        # Debug variables
        self.debug = debug

        self.logger.info("Robot is initialized and ready to start.")

    def _refresh_handler(self, node) -> None:
        """
        This method is executed everytime that the variables of the robot are refreshed. Here are executed important
        tasks such as fall checking and odometry.

        :param node: Node id of the robot. This is passed by the refresh event.
        """
        # Calculate variation in time between refreshes
        now = time.time_ns()
        dt = now - self.prev_time

        # Gets data and recalculates pose through Odometry
        left_speed, right_speed = self.get_motors_speed()
        self._recalculate_odometry(left_speed, right_speed, dt / 1e9, kd=0.325387)

        # Add speeds to the speed buffer
        self.last_speeds.append((now, left_speed, right_speed))

        # Handles the case where the robot is about to fall
        if self.robot[self.node]["prox.ground.delta"][0] < 100 or self.robot[self.node]["prox.ground.delta"][1] < 100:
            self.set_motors_speed(0, 0)
            self.current_mode = Modes.STOPPED

        # Debug message
        #if self.debug:
        #    self.logger.debug(f"Robot variables refreshed! Time since last refresh -> {dt/1e6}ms")

        # End of the method, declares variables for the next refresh
        self.prev_time = now

    def _recalculate_odometry(self, left_speed: int, right_speed: int, dt: float, *, kd: float = 0.033):
        """
        This method updates robot odometry information.

        :param left_speed: Read speed of the left wheel of the robot.
        :param right_speed: Read speed of the right wheel of the robot.
        :param dt: Time span in seconds since last calculation.
        :param kd: Calibrated constant in mm/V*s (where V is velocity from the robot, s is seconds, and mm is millimeters)
        """
        # Gets current pose angle
        angle = self.pose.get_angle()

        # Calculates distances traveled by left and right wheels and the center of the robot
        dl = left_speed * dt * kd
        dr = right_speed * dt * kd
        dc = (dr + dl) / 2

        # Calculates displacement on the new pose
        dth = (dl - dr) / self.L
        dy = dc * math.sin(dth)
        dx = dc * math.cos(dth)

        # Adds each variation to the current x, y, angle values
        self.pose.add_pose_variation(dx, dy, dth)

        # Debug message
       ## if self.debug:
        #    self.logger.debug(f"Odometry -> dx: {round(dx, 3)}, dy: {round(dy, 3)}, dth: {round(math.degrees(dth), 3)}, {self.pose}")

    # Below methods are robot extra utilities to used with ThymioDirect library
    def set_motors_speed(self, left: int, right: int, *, min_vel: int = -500, max_vel: int = 500, dead_band: int = 0) -> (int, int):
        """
        Updates robot target speed for the wheels. This method implements speed saturation and dead band in turn of 0.

        :param left: Speed target to the right wheel.
        :param right: Speed target to the left wheel.
        :param min_vel: Default is -500. Lowest possible speed.
        :param max_vel: Default is 500. Highest possible speed.
        :param dead_band: Default is 0. This acts like a dead band, not allowing values smaller than |dead_band|.

        :return: A tuple containing left speed and right speed targets, respectively.
        """
        # Saturates sent speed to the range [-500; 500]
        left_speed = max(min(left, max_vel), min_vel)
        right_speed = max(min(right, max_vel), min_vel)

        # Nulls speeds in the dead band (if configured one) and updates the robot
        self.robot[self.node]["motor.left.target"] = int(left_speed) if abs(left_speed) > dead_band else 0
        self.robot[self.node]["motor.right.target"] = int(right_speed) if abs(right_speed) > dead_band else 0

        # Debug message
        if self.debug:
            self.logger.debug(f"Motors speed set to -> {left_speed}, {right_speed}")
            self.logger.debug({f"Left: {left}, right: {right}"})

        return left_speed, right_speed

    def get_motors_target(self) -> (int, int):
        """
        Gets the motor target from the robot. This method implements a method because in robot memory
        negative numbers start from 65535..65035 (specifically for a range from -500 to 0).

        :return: A tuple containing left speed and right speed targets, respectively.
        """
        if self.robot[self.node]["motor.left.target"] > 500:
            # Maps [65535; 65035[ to [-500; 0[ when value read from robot greater than 500
            left_target = translate(self.robot[self.node]["motor.left.target"], 65035, 65535, 0, -500)
        else:
            left_target = self.robot[self.node]["motor.left.target"]

        if self.robot[self.node]["motor.right.target"] > 500:
            # Maps [65535; 65035[ to [-500; 0[ when value read from robot greater than 500
            right_target = translate(self.robot[self.node]["motor.right.target"], 65035, 65535, 0, -500)
        else:
            right_target = self.robot[self.node]["motor.right.target"]

        return left_target, right_target

    def get_motors_speed(self) -> (int, int):
        """
        Gets the motor speed readings from the robot. This method implements a method because in robot memory
        negative numbers start from 65535..65035 (specifically for a range from -500 to 0), on top of that this map only
        starts being done after 700 to let readings greater than 500 pass.

        :return: A tuple containing left speed and right speed, respectively.
        """
        # Checks motors speed targets
        left_target, right_target = self.get_motors_target()

        # If target is 0 than return also 0 (to remove noise)
        if not left_target == 0:
            if self.robot[self.node]["motor.left.speed"] > 700:
                # Maps [65535; 65035[ to [-500; 0[ when value read from robot greater than 700
                left_speed = translate(self.robot[self.node]["motor.left.speed"], 65035, 65535, -500, 0)
            else:
                if self.robot[self.node]["motor.left.speed"] < 0:
                    self.logger.warning(f"Weird speed value from Thymio robot! motor.left.speed -> {self.robot[self.node]['motor.left.speed']}")
                left_speed = self.robot[self.node]["motor.left.speed"]
        else:
            left_speed = 0

        # If target is 0 than return also 0 (to remove noise)
        if not right_target == 0:
            if self.robot[self.node]["motor.right.speed"] > 700:
                # Maps [65535; 65035[ to [-500; 0[ when value read from robot greater than 700
                right_speed = translate(self.robot[self.node]["motor.right.speed"], 65035, 65535, -500, 0)
            else:
                if self.robot[self.node]["motor.right.speed"] < 0:
                    self.logger.warning(f"Weird speed value from Thymio robot! motor.right.speed -> {self.robot[self.node]['motor.right.speed']}")
                right_speed = self.robot[self.node]["motor.right.speed"]
        else:
            right_speed = 0
        return left_speed, right_speed

    # Below code used in the robots main logic
    def _is_at_target(self, point: Point, threshold: float) -> bool:
        """
        Checks whether the robot is at the given point, within a margin of error.

        :param point: A destiny point to compare the robot's pose against.
        :param threshold: The acceptable margin of error of the given point.

        :return: True is at the point within threshold, False otherwise.
        """
        return abs(point.get_x() - self.pose.get_x()) < threshold and abs(point.get_y() - self.pose.get_y()) < threshold

    def _visual_servoring_behavior(self, d_pid: PIDController, th_pid: PIDController, face: tuple) -> None:
        """
        Handles visual servoring. This method is supposed to run whenever data from the vision processor is received.

        :param d_pid: PIDController instance to control distance from the face/target.
        :param th_pid: PIDController instance to control alignment with the face/target.
        :param face: A tuple containing face x coordinate and face distance, respectively.
        """
        x, distance = face

        # Handling situation where visual servoring does not detect any face
        if (x is None or distance is None):
            if not self.get_motors_target() == (0, 0):
                self.set_motors_speed(0, 0)
                self.logger.warning(f"No face being detected, stopping motors!")
            return

        d_out = d_pid.sample(4 - distance) # distancia a que o robo esta da cara da pessoa
        th_out = th_pid.sample(x)

        self.logger.debug(f"D Out: {d_out}, thout: {th_out}")

        # Assign robot velocities
        l_speed = d_out * math.exp(-1/100 * abs(x)) - th_out
        r_speed = d_out * math.exp(-1/100 * abs(x)) + th_out

        self.logger.debug(f"l_speed: {l_speed}, r_speed: {r_speed}\n")
        self.set_motors_speed(l_speed, r_speed, dead_band=50)

    async def _handle_vision(self, vision_pipe: multiprocessing.Queue, intermediate_queue: asyncio.Queue) -> None:
        """
        This asyncio coroutine handles data coming to the vision_pipe. This is executed when remote_vision is false.

        :param vision_pipe: The pipeline from the local vision processor instance.
        :param intermediate_queue: The pipeline to send the data to adequate handler.
        """
        while True:
            # Gets data from the pipe [TERMINATES LOOP IF TIMEOUT EXPIRES]
            try:
                frame, face_x, face_y, face_size, hand_gesture = vision_pipe.get(block=True, timeout=2)
                await intermediate_queue.put((face_x, face_y, face_size, hand_gesture))

                cv2.imshow('Face Tracking', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                await asyncio.sleep(0.01)
            except queue.Empty as e:
                self.set_motors_speed(0, 0)
                self.logger.critical(e)
                asyncio.get_event_loop().stop()

    async def _handle_vision_remote(self, intermediate_queue: asyncio.Queue) -> None:
        """
        This asyncio coroutine handles data coming from a remote publisher. This is executed when remote_vision is True.

        :param intermediate_queue: The pipeline to send the data to adequate handler.
        """
        # Create 0MQ client/subscriber to communicate with vision server
        ctx = aiozmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribes to all messages
        sock.connect(f"tcp://{self.remote_addr}")

        while True:
            self.logger.debug("Waiting for data from the vision publisher.")
            data = await sock.recv_string()
            self.logger.debug("Vision processed data received from the server!")
            face_x, face_size, hand_gesture = json.loads(data)
            self.logger.debug(f"Face data received {json.loads(data)}")
            await intermediate_queue.put((face_x, face_size, hand_gesture))
            await asyncio.sleep(0.005)

    async def _handle_robot(self, intermediate_queue: asyncio.Queue) -> None:
        """
        This asyncio coroutine handles data received by vision handlers. Interprets data and acts based on it.

        :param intermediate_queue: The pipeline to receive data from vision handlers.
        """
        # Visual Servoring PID's
        vs_d_pid = PIDController(0, 900, 0, 300).set_saturation(400, -400)
        vs_th_pid = PIDController(0, 1, 0, 0.5).set_saturation(400, -400)
        while True:
            # Gets processed data from the vision handler task
            data = await intermediate_queue.get()
            face_x, face_size, hand_gesture = data

            # Changes program mode on hand gesture from the vision
            if (hand_gesture == "PALM") and self.current_mode not in [Modes.STOPPED, Modes.RETURN]:
                self.logger.info("'PALM detected! Stopping the robot.'")
                self.current_mode = Modes.STOPPED
            elif hand_gesture == "THUMBS_UP" and self.current_mode not in [Modes.VISUAL_SERVORING, Modes.RETURN, Modes.MOVING]:
                self.logger.info("'THUMBS_UP' detected! Entering companion/follower mode.")
                self.current_mode = Modes.VISUAL_SERVORING
                vs_d_pid.reset()
                vs_th_pid.reset()
            elif hand_gesture == "PIECE_SIGN" and self.current_mode not in [Modes.RETURN, Modes.MOVING]:
                self.logger.info("'PIECE_SIGN' detected! Returning to the start.")
                self.current_mode = Modes.RETURN

            # Make decisions based on the current mode
            if self.current_mode == Modes.STOPPED:
                self.set_motors_speed(0, 0)
            elif self.current_mode == Modes.VISUAL_SERVORING:
                self._visual_servoring_behavior(vs_d_pid, vs_th_pid, (face_x, face_size))
            elif self.current_mode == Modes.RETURN:
                start = Point(0, 0)
                loop = asyncio.get_event_loop()
                self.current_mode = Modes.MOVING
                move_task = loop.create_task(self._move_to_point(start))
                move_task.add_done_callback(lambda x: setattr(self, "current_mode", Modes.STOPPED))

            # A small sleep to force the task to yield at least once
            await asyncio.sleep(0.001)

    async def _move_to_point(self, point: Point) -> None:
        """
        This asyncio coroutine allows run move_to_point loop without blocking the rest of the operations. This method
        controls the robot to run from its current point to the given point, using interpolation to create multiple
        waypoints. The robot is then controlled to reach all the points until the final point.

        **Note:** If the robot mode changes from MOVING to any other mode during the process, the loop will
        automatically stop moving.

        :param point: Destiny point.
        """
        # Interpolate waypoints between robot pose and the destiny
        waypoints = self.pose.interpolate_waypoints_to(point, 20)

        # Info message
        self.logger.info(f"Moving from -> {self.pose} to -> {point}")

        # Loop through waypoints and try to follow them
        for point in waypoints:
            # Create PID controllers for movement and orientation
            d_pid = PIDController(0, 250, 1 / 100, 30).set_saturation(500, 0)
            th_pid = PIDController(0, 260, 0, 80).set_saturation(500, -500)

            # Loop the control until robot arrives to waypoint
            while not self._is_at_target(point, self.pos_threshold):
                # Handles the case where the robot current mode is changed by other task
                if self.current_mode != Modes.MOVING:
                    return

                # A small sleep to force the task to yield at least once
                await asyncio.sleep(0.005)

                # Angle error (for th_pid) and distance (position error for d_pid)
                angle_e = ((self.pose.angle_with(point) - self.pose.get_angle()) + math.pi) % (2 * math.pi) - math.pi
                distance_e = -self.pose.distance_from(point)

                # Updating PID controllers with respective inputs
                d_out = d_pid.sample(distance_e)
                th_out = th_pid.sample(angle_e)

                # Calculating speed with PID outputs (distance outputs is weighted based on angle error)
                l_speed = d_out * math.exp(-18 * abs(angle_e)) - th_out
                r_speed = d_out * math.exp(-18 * abs(angle_e)) + th_out

                # Setting the motors speed with a dead band of 50 (from -50 to 50)
                self.set_motors_speed(l_speed, r_speed, dead_band=50)
            self.logger.info(f"Arrived at waypoint -> {point}")

        # Stop at final waypoint arriving
        self.set_motors_speed(0, 0)
        if self._is_at_target(point, self.pos_threshold):
            self.logger.info(f"Arrived to final point -> {self.pose}")

    def run_loop(self) -> None:
        """
        Create the asyncio loop and starts all the relevant tasks to the logic. This is the entry point to the robot
        main logic and control.
        """
        intermediate_queue = asyncio.Queue()
        vision_pipe = multiprocessing.Queue()
        loop = asyncio.get_event_loop()
        if self.remote_vision:
            loop.create_task(self._handle_vision_remote(intermediate_queue))
        else:
            loop.create_task(self._handle_vision(vision_pipe, intermediate_queue))
        loop.create_task(self._handle_robot(intermediate_queue))
        loop.run_forever()

    def move_to_point(self, point: Point) -> None:
        """
        This method is an asyncio wrapper to the _move_to_point() coroutine. It allows this function to be executed
        synchronously outside of asyncio coroutines.

        :param point: Destiny point.
        """
        self.current_mode = Modes.MOVING
        asyncio.run(self._move_to_point(point))
        self.current_mode = Modes.STOPPED

    # Below methods used when trying to get and calculates speeds and averages (USED FOR ODOMETRY CALIBRATION PURPOSES)
    def _motors_speed_average_between(self, start_time, end_time) -> (float, float):
        # Filter buffered speeds in deque between start_time and end_time
        filtered_speeds = [
            (timestamp, left_speed, right_speed)
            for timestamp, left_speed, right_speed in self.last_speeds
            if start_time < timestamp < end_time
        ]

        # Return 0 averages if there was no speed in the buffer in the time period
        if not filtered_speeds:
            return 0, 0

        # Calculate mean speeds
        mean_left_speed = sum(speed[1] for speed in filtered_speeds) / len(filtered_speeds)
        mean_right_speed = sum(speed[2] for speed in filtered_speeds) / len(filtered_speeds)

        return mean_left_speed, mean_right_speed

    def _movement_with_speed_average(self, vel: int, t: float):
        start = time.time_ns()
        self.set_motors_speed(vel, vel)
        time.sleep(t)
        end = time.time_ns()
        self.set_motors_speed(0, 0)
        avg = self._motors_speed_average_between(start, end)
        delta_t = end - start
        print(f"Average Speed: {avg}\tElapsed Time: {delta_t}ms")

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--server_addr", help="Zmq server socket address (e.g 127.0.0.1:5555)", required=True)
    parser.add_argument("--debug", help="Enables debug messages (a lots of them)", action="store_true", required=False)
    args = parser.parse_args()

    # Set up logging configuration
    debug_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=debug_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("client")

    robot = ThymioRobot(93.5, refreshing_rate=0.01, debug=args.debug, remote_vision=True, remote_addr=args.server_addr)
    robot.run_loop()