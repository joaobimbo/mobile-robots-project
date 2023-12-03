import time

class PIDController:

    def __init__(self, sp: float, kp: float, ki: float, kd: float):
        # Controller specific variables
        self.sp = sp
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Controller characteristics
        self.lower = None
        self.upper = None
        self.integral_limit = None

        # Controller specific auxiliary variables
        self.err_current = 0
        self.err_previous = 0
        self.err_integral = 0
        self.time_previous = time.time_ns()

    def set_sp(self, sp: float):
        self.sp = sp
        return self

    def controller_threshold(self, upper: float, lower: float) -> (float, float):
        self.upper = upper
        self.lower = lower
        return self.upper, self.lower

    def integral_saturation(self, sat: float) -> float:
        self.integral_limit = sat
        return self.integral_limit

    def integral_reset(self) -> float:
        self.err_integral = 0
        return self.err_integral

    def _add_integral(self, err: float, sample_time: float = 1) -> float:
        if self.integral_limit is not None:
            if self.err_integral + err > self.integral_limit:
                return self.integral_limit
            elif self.err_integral + err < -1 * self.integral_limit:
                return -1 * self.integral_limit
        return self.err_integral + err * sample_time

    def sample(self, input: float):
        # Get current time and calculate elapsed time
        time_current = time.time_ns()
        delta_time = (time_current - self.time_previous) / 1e9

        # Calculate error, error derivative and error integral
        self.err_current = self.sp - input
        err_derivative = (self.err_current - self.err_previous)/delta_time if delta_time != 0 else 0
        self.err_integral = self._add_integral(self.err_current, delta_time)

        # Calculate the controller output
        output = self.kp * self.err_current + self.ki * self.err_integral + self.kd * err_derivative

        # Check if output goes over saturation value
        return max(min(output, self.upper), self.lower)

    def set_upper(self, upper: int):
        self.upper = upper

    def set_lower(self, lower: int):
        self.lower = lower

    def set_saturation(self, upper: int, lower: int):
        self.set_upper(upper)
        self.set_lower(lower)
        return self

    def reset(self):
        self.err_current = 0
        self.err_previous = 0
        self.err_integral = 0
        self.time_previous = time.time_ns()
