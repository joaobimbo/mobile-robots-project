import asyncio
import math
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

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)