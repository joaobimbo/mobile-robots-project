import math
import time
from collections import deque
from enum import Enum
from thymiodirect import Connection
from thymiodirect import Thymio
from pid import PIDController
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Modes(Enum):
    VISUAL_SERVORING = 1
    STOPPED = 2
    RETURNING = 3
    AVOIDING = 4
    EMERGENCY_STOP = 5

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def update_position(self, x: float, y: float) -> (float, float):
        self.x = x
        self.y = y
        return self.x, self.y

    def get_position(self) -> (float, float):
        return self.x, self.y

    def get_x(self) -> float:
        return self.x

    def get_y(self) -> float:
        return self.y

    def distance_from(self, point: 'Point'):
        return math.sqrt(math.pow(point.get_x() - self.x, 2) + math.pow(point.get_y() - self.y, 2))

    def angle_with(self, point: 'Point', *, degrees: bool = None):
        if degrees:
            return math.degrees(math.atan2(point.get_x() - self.get_x(), point.get_y() - self.get_y()))
        return math.atan2(point.get_y() - self.get_y(), point.get_x() - self.get_x())

    def __str__(self):
        return f"Point({round(self.x, 2)}mm, {round(self.y, 2)}mm)"

class Pose(Point):
    def __init__(self, x: float, y: float, angle: float):
        super().__init__(x, y)
        self.angle = angle

    def update_pose(self, x: float, y: float, angle: float) -> (float, float, float):
        super().update_position(x, y)
        self.angle = angle % (2 * math.pi)
        return super().get_x(), super().get_y(), self.angle

    def get_pose(self, *, degrees: bool | None = None) -> (float, float, float):
        return super().get_x(), super().get_y(), self.angle

    def get_angle(self, *, degrees: bool | None = None) -> float:
        if degrees:
            return math.degrees(self.angle)
        return self.angle

    def get_normalized_angle(self, *, degrees: bool | None = None) -> float:
        normalized_angle = (self.angle + math.pi) % (2 * math.pi) - math.pi
        if degrees:
            return math.degrees(normalized_angle)
        return normalized_angle

    def add_pose_variation(self, dx: float, dy: float, dth: float):
        self.update_pose(self.x+dx, self.y+dy, self.angle+dth)

    def interpolate_waypoints_to(self, point: Point, chunk_size: float) -> list:
        waypoints = []
        distance = self.distance_from(point)

        # Number of waypoints to create, with 1cm intervals
        num_waypoints = int(distance / chunk_size)

        for i in range(1, num_waypoints + 1):
            ratio = i / num_waypoints
            waypoints.append(Point(
                self.x + ratio * (point.get_x() - self.x),
                self.y + ratio * (point.get_y() - self.y)
            ))

        return waypoints

    def __str__(self):
        return f"Pose({round(self.x, 2)}mm, {round(self.y, 2)}mm, {round(self.get_angle(degrees=True), 2)}º)"

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

    def set_motors_speed(self, left: int, right: int) -> (int, int):
        left_speed = max(min(left, 500), -500)
        right_speed = max(min(right, 500), -500)
        self.robot[self.node]["motor.left.target"] = int(left_speed)
        self.robot[self.node]["motor.right.target"] = int(right_speed)

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

        # If target ==
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

    def move_to_point(self, point: Point):
        # Interpolate waypoints between robot pose and the destiny
        waypoints = robot.pose.interpolate_waypoints_to(point, 20)
        for point in waypoints:
            print(point)
        # Info message
        self.logger.info(f"Moving from -> {self.pose} to -> {point}")

        # Flag to stop distance control when robot is WAYYYY out of the correct direction
        at_angle = False

        # Loop through waypoints and try to follow them
        for point in waypoints:
            # Create PID controllers for movement and orientation
            d_pid = PIDController(0, 100, 1 / 100, 50).set_saturation(250, 0)
            th_pid = PIDController(0, 200, 1/300, 200).set_saturation(400, -400)

            # Loop the control until robot arrives to waypoint
            while not self._is_at_target(point, self.pos_threshold):
                time.sleep(0.01)

                # Angle error (for th_pid) and distance (position error for d_pid)
                angle_e = self.pose.angle_with(point) - self.pose.get_normalized_angle()
                distance_e = -self.pose.distance_from(point)

                # Enable 'at_flag' when angle error is acceptable
                if (not at_angle) and abs(angle_e) < self.angle_threshold_lower:
                    self.logger.info(f"Acceptable orientation -> error of {math.degrees(angle_e)} degrees")
                    at_angle = True

                # Disables 'at_flag' when error is unacceptable
                if at_angle and abs(angle_e) > self.angle_threshold_upper:
                    self.logger.warning(f"Unacceptable orientation -> error of {math.degrees(angle_e)} degrees")
                    at_angle = False

                # Updating PID controllers with respective inputs
                d_out = d_pid.sample(distance_e) if at_angle else 0
                th_out = th_pid.sample(angle_e)

                # Assign robot velocities
                l_speed = d_out - th_out
                r_speed = d_out + th_out

                self.set_motors_speed(l_speed, r_speed)
            self.logger.info(f"Arrived at waypoint -> {point}")

        # Stop at final waypoint arriving
        self.set_motors_speed(0, 0)
        self.logger.info(f"Arrived to final point -> {self.pose}")

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


if __name__ == "__main__":
    robot = ThymioRobot(93.5)
    robot.set_motors_speed(0, 0)
    try:
        robot.move_to_point(Point(130, 90))
    except:
        robot.set_motors_speed(0, 0)

