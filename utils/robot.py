import asyncio
import math
import time
from collections import deque
from enum import Enum
from multiprocessing import Queue

import cv2
from thymiodirect import Connection
from thymiodirect import Thymio
from utils.pid import PIDController
from utils.auxils import translate, Point, Pose
import logging
import queue

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Modes(Enum):
    VISUAL_SERVORING = 1
    STOPPED = 2
    RETURN = 3
    MOVING = 4
    AVOIDING = 5
    EMERGENCY_STOP = 6

class ThymioRobot:
    def __init__(self, L: float, *, refreshing_rate: float = 0.01, refreshing_coverage: dict | list = None, debug: bool = False):
        # Setting up logger for the ThymioRobot
        self.logger = logging.getLogger('ThymioRobot')

        # Setup Thymio connection
        port = Connection.serial_default_port()
        self.robot = Thymio(serial_port=port,
                    refreshing_rate=refreshing_rate,
                    refreshing_coverage=refreshing_coverage,
                    on_connect=lambda node_id: self.logger.info(f"ThymioRobot connected successfully on node -> {node_id}"))
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

        # Debug variables
        self.debug = debug

    def _refresh_handler(self, node):
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
        if self.debug:
            self.logger.debug(f"Robot variables refreshed! Time since last refresh -> {dt/1e6}ms")

        # End of the method, declares variables for the next refresh
        self.prev_time = now

    def _recalculate_odometry(self, left_speed: int, right_speed: int, dt: float, *, kd: float = 0.033):
        # Gets current pose angle
        angle = self.pose.get_angle()

        # Calculates distances traveled by left and right wheels and the center of the robot
        dl = left_speed * dt * kd
        dr = right_speed * dt * kd
        dc = (dr + dl) / 2

        # Calculates displacement on the new pose
        dth = (dl - dr) / self.L
        dy = dc * math.sin(angle)
        dx = dc * math.cos(angle)

        # Adds each variation to the current x, y, angle values
        self.pose.add_pose_variation(dx, dy, dth)

        # Debug message
        if self.debug:
            self.logger.debug(f"Odometry -> dx: {round(dx, 3)}, dy: {round(dy, 3)}, dth: {round(math.degrees(dth), 3)}, {self.pose}")

    def set_motors_speed(self, left: int, right: int, *, min_vel: int = -500, max_vel: int = 500, dead_band: int = 0) -> (int, int):
        left_speed = max(min(left, max_vel), min_vel)
        right_speed = max(min(right, max_vel), min_vel)
        self.robot[self.node]["motor.left.target"] = int(left_speed) if abs(left_speed) > dead_band else 0
        self.robot[self.node]["motor.right.target"] = int(right_speed) if abs(right_speed) > dead_band else 0

        # Debug message
        if self.debug:
            self.logger.debug(f"Motors speed set to -> {left_speed}, {right_speed}")

        return left_speed, right_speed

    def get_motors_target(self) -> (int, int):
        if self.robot[self.node]["motor.left.target"] > 500:
            left_target = translate(self.robot[self.node]["motor.left.target"], 65035, 65535, 0, -500)
        else:
            left_target = self.robot[self.node]["motor.left.target"]

        if self.robot[self.node]["motor.right.target"] > 500:
            right_target = translate(self.robot[self.node]["motor.right.target"], 65035, 65535, 0, -500)
        else:
            right_target = self.robot[self.node]["motor.right.target"]

        return left_target, right_target

    def get_motors_speed(self) -> (int, int):
        # Checks motors speed targets
        left_target, right_target = self.get_motors_target()

        # If target is 0 than return also 0 (to remove noise)
        if not left_target == 0:
            # This 700 is a 200 velocity margin because occasionally speed measurements go higher than 500
            if self.robot[self.node]["motor.left.speed"] > 700:
                left_speed = translate(self.robot[self.node]["motor.left.speed"], 65035, 65535, -500, 0)
            else:
                if self.robot[self.node]["motor.left.speed"] < 0:
                    print("Weird value! ", self.robot[self.node]["motor.right.speed"])
                left_speed = self.robot[self.node]["motor.left.speed"]
        else:
            left_speed = 0

        # If target is 0 than return also 0 (to remove noise)
        if not right_target == 0:
            # This 700 is a 200 velocity margin because occasionally speed measurments go higher than 500
            if self.robot[self.node]["motor.right.speed"] > 700:
                right_speed = translate(self.robot[self.node]["motor.right.speed"], 65035, 65535, -500, 0)
            else:
                if self.robot[self.node]["motor.right.speed"] < 0:
                    print("Weird value! ", self.robot[self.node]["motor.right.speed"])
                right_speed = self.robot[self.node]["motor.right.speed"]
        else:
            right_speed = 0
        return left_speed, right_speed

    def _is_at_target(self, point: Point, threshold: float) -> bool:
        return abs(point.get_x() - self.pose.get_x()) < threshold and abs(point.get_y() - self.pose.get_y()) < threshold

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

    def _visual_servoring_behavior(self, d_pid: PIDController, th_pid: PIDController, face: tuple):
        x, y, size = face

        # Handling situation where visual servoring does not detect any face
        if x is None or size is None:
            self.set_motors_speed(0, 0)
            return self.logger.warning(f"No face being detected, stopping motors!")

        d_out = d_pid.sample(4 - size)
        th_out = th_pid.sample(x)

        # Assign robot velocities
        l_speed = d_out * math.exp(-1/100 * abs(x)) - th_out
        r_speed = d_out * math.exp(-1/100 * abs(x)) + th_out

        self.set_motors_speed(l_speed, r_speed, dead_band=50)

    async def _handle_vision(self, vision_pipe: Queue, intermediate_queue: asyncio.Queue):
        while True:
            # Gets data from the pipe [TERMINATES LOOP IF TIMEOUT EXPIRES]
            try:
                frame, face_x, face_y, face_size, hand_gesture = vision_pipe.get(block=True, timeout=2)
                await intermediate_queue.put((face_x, face_y, face_size, hand_gesture))

                cv2.imshow('Face Tracking', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                await asyncio.sleep(0.001)

                await asyncio.sleep(0.01)
            except queue.Empty as e:
                self.logger.critical(e)

    async def _handle_robot(self, intermediate_queue: asyncio.Queue):
        # Visual Servoring PID's
        vs_d_pid = PIDController(0, 700, 0, 300).set_saturation(400, -400)
        vs_th_pid = PIDController(0, 1, 0, 0).set_saturation(400, -400)
        while True:
            # Gets processed data from the vision handler task
            data = await intermediate_queue.get()
            face_x, face_y, face_size, hand_gesture = data

            # Changes program mode on hand gesture from the vision
            if hand_gesture == "PALM" and self.current_mode not in [Modes.STOPPED, Modes.RETURN]:
                self.current_mode = Modes.STOPPED
            elif hand_gesture == "THUMBS_UP" and self.current_mode not in [Modes.VISUAL_SERVORING, Modes.RETURN, Modes.MOVING]:
                self.current_mode = Modes.VISUAL_SERVORING
                vs_d_pid.reset()
                vs_th_pid.reset()
            elif hand_gesture == "PIECE_SIGN" and self.current_mode not in [Modes.RETURN, Modes.MOVING]:
                self.current_mode = Modes.RETURN

            # Make decisions based on the current mode
            if self.current_mode == Modes.STOPPED:
                self.set_motors_speed(0, 0)
            elif self.current_mode == Modes.VISUAL_SERVORING:
                self._visual_servoring_behavior(vs_d_pid, vs_th_pid, (face_x, face_y, face_size))
            elif self.current_mode == Modes.RETURN:
                start = Point(0, 0)
                loop = asyncio.get_event_loop()
                self.current_mode = Modes.MOVING
                move_task = loop.create_task(self._move_to_point(start))
                move_task.add_done_callback(lambda x: setattr(self, "current_mode", Modes.STOPPED))

            # A small sleep to force the task to yield at least once
            await asyncio.sleep(0.001)

    async def _move_to_point(self, point: Point):
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

    def run_loop(self, vision_pipe: Queue):
        intermediate_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        loop.create_task(self._handle_vision(vision_pipe, intermediate_queue))
        loop.create_task(self._handle_robot(intermediate_queue))
        loop.run_forever()

    def move_to_point(self, point: Point):
        self.current_mode = Modes.MOVING
        asyncio.run(self._move_to_point(point))
        self.current_mode = Modes.STOPPED



if __name__ == "__main__":
    robot = ThymioRobot(93.5, debug=False)
    robot.set_motors_speed(0, 0)
    try:
        """robot.move_to_point(Point(200, 0))
        robot.move_to_point(Point(200, -100))
        robot.move_to_point(Point(0, -100))
        robot.move_to_point(Point(0, 0))"""
        robot.move_to_point(Point(200, 0))
    except:
        robot.set_motors_speed(0, 0)