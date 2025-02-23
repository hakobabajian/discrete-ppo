import krpc
import time
from timeit import default_timer


class Environment:
    def __init__(self, time_step=0.005, control_step=0.02, actions=3, observations=4, cruise_speed=80,
                 cruise_acceleration=150, target_quantity=100, target_offset=15, max_runtime=60):
        conn = krpc.connect()
        self.space_center = conn.space_center
        self.vessel = self.space_center.active_vessel
        self.ref_frame = self.space_center.ReferenceFrame.create_hybrid(
            position=self.vessel.orbit.body.reference_frame,
            rotation=self.vessel.surface_reference_frame)
        self.time_step = time_step
        self.t = time_step
        self.control_step = control_step
        self.actions = actions
        self.observation_space_shape = (observations,)
        self.quantity_name = "mean_altitude"
        self.control_name = "pitch"
        self.cruise_speed = cruise_speed
        self.cruise_acceleration = cruise_acceleration
        self.target_quantity = target_quantity
        self.target_offset = target_offset
        self.initial_speed = 0
        self.start_time = default_timer()
        self.step_start = default_timer()
        self.max_runtime = max_runtime
        self.max_punishment = -1 * max_runtime / time_step * abs(target_offset - target_quantity)

    def reset_vessel(self):
        conn = krpc.connect()
        self.space_center = conn.space_center
        self.vessel = self.space_center.active_vessel
        self.ref_frame = self.space_center.ReferenceFrame.create_hybrid(
            position=self.vessel.orbit.body.reference_frame,
            rotation=self.vessel.surface_reference_frame)

    # Helper methods
    def get_initial_derivatives(self, quantity_name, n):
        derivatives_matrix = []
        zeroth_derivatives = []

        for i in range(n + 1):
            zeroth_derivatives.append(getattr(self.vessel.flight(self.ref_frame), quantity_name))
            time.sleep(self.time_step)
        derivatives_matrix.append(zeroth_derivatives)

        for derivatives in derivatives_matrix:
            if len(derivatives) == 1:
                break
            next_derivatives = []
            for i, derivative in enumerate(derivatives):
                if i < len(derivatives) - 1:
                    next_derivatives.append((derivatives[i + 1] - derivatives[i]) / self.time_step)
            derivatives_matrix.append(next_derivatives)

        initial_derivatives = [sum(derivatives) / len(derivatives) for derivatives in derivatives_matrix]
        return initial_derivatives

    def get_time_derivative(self, quantity_name, quantity_before_step):
        quantity_current = getattr(self.vessel.flight(self.ref_frame), quantity_name)
        instantaneous_time_derivative = (quantity_current - quantity_before_step) / self.t
        return instantaneous_time_derivative, quantity_current

    def control_quantity(self, quantity_name, quantity_before_step, quantity_bound, time_derivative_bound,
                         control_name):
        time_derivative, quantity_current = self.get_time_derivative(quantity_name, quantity_before_step)
        if quantity_current < quantity_bound and time_derivative < time_derivative_bound:
            setattr(self.vessel.control, control_name, getattr(self.vessel.control, control_name) + self.control_step)
        else:
            setattr(self.vessel.control, control_name, getattr(self.vessel.control, control_name) - self.control_step)
        return quantity_current

    def get_time_derivatives(self, quantity_name, initial_derivatives):
        final_derivatives = [getattr(self.vessel.flight(self.ref_frame), quantity_name)]
        for initial_derivative in initial_derivatives:
            final_derivatives.append((final_derivatives[-1] - initial_derivative) / self.t)
        return final_derivatives

    def get_reward(self, quantity_name):
        quantity = getattr(self.vessel.flight(self.ref_frame), quantity_name)
        if quantity > self.target_quantity:
            reward = -1 * quantity + self.target_quantity + self.target_offset
        else:
            reward = quantity - self.target_quantity + self.target_offset
        return int(reward) + 1

    # Resets environment after termination
    def reset(self):
        self.space_center.revert_to_launch()
        self.reset_vessel()
        time.sleep(2)
        self.vessel.control.activate_next_stage()
        self.initial_speed = 0
        self.start_time = default_timer()
        initial_altitude_derivatives = self.get_initial_derivatives(self.quantity_name, self.observation_space_shape[0] - 2)
        first_observation = [self.initial_speed] + initial_altitude_derivatives
        return first_observation

    # Takes one time step forward and executes action policy from network, returns previous policy's outcome
    def step(self, action, previous_observation):
        elapsed_time = default_timer() - self.step_start
        if elapsed_time >= self.time_step:
            self.t = elapsed_time
        else:
            time.sleep(self.time_step - elapsed_time)
            self.t = self.time_step
        self.initial_speed = self.control_quantity("speed", self.initial_speed, self.cruise_speed,
                                                   self.cruise_acceleration, "throttle")
        if action == 0:
            setattr(self.vessel.control, self.control_name,
                    getattr(self.vessel.control, self.control_name) + self.control_step)
        elif action == 1:
            pass
        elif action == 2:
            setattr(self.vessel.control, self.control_name,
                    getattr(self.vessel.control, self.control_name) - self.control_step)

        previous_altitude_derivatives = previous_observation[1:]
        altitude_derivatives = self.get_time_derivatives(self.quantity_name, previous_altitude_derivatives)[:-1]
        next_observation = [self.initial_speed] + altitude_derivatives

        reward = self.get_reward(self.quantity_name)
        if self.vessel.crew_count == 0 or self.vessel.situation.name == "splashed":
            print("Environment Terminated: Vessel Inoperable")
            reward += self.max_punishment
            done = True
        elif altitude_derivatives[0] > 5 * self.target_quantity:
            print("Environment Terminated: Out of bounds")
            reward += self.max_punishment
            done = True
        elif default_timer() - self.start_time > self.max_runtime:
            print(f"Environment Terminated: Time out {self.max_runtime}s")
            if self.vessel.situation.name == "landed":
                reward += self.max_punishment
            done = True
        else:
            done = False
        self.step_start = default_timer()
        return next_observation, reward, done
