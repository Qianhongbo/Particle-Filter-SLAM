import numpy as np
from pr2_utils import read_data_from_csv
import matplotlib.pyplot as plt

class trajectory:
    def __init__(self, resolution, left_wheel_diameter, right_wheel_diameter, wheel_base):
        self.encoder_timestamps, self.encoder_data, self.fog_timestamps, self.fog_data = self.load_data()
        self.resolution = resolution
        self.left_diameter = left_wheel_diameter
        self.right_diameter = right_wheel_diameter
        self.wheel_base = wheel_base
        self.tra = np.empty([self.encoder_data.shape[0] * 10, 2])

    def load_data(self):
        """
        read data from "lidar.csv" and "fog.csv"
        """
        encoder_timestamps, encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')
        fog_timestamps, fog_data = read_data_from_csv('data/sensor_data/fog.csv')
        return encoder_timestamps, encoder_data, fog_timestamps, fog_data

    def compute_trajectory(self):
        v, omega, j = 0, 0, 1 # initialize
        state = np.array((0,0,0), dtype = "float64")
        multiple = self.fog_timestamps.shape[0] // self.encoder_timestamps.shape[0]
        for i in range(1, self.encoder_timestamps.shape[0] * multiple):
            t_fog = (self.fog_timestamps[i] - self.fog_timestamps[i - 1]) / (10 ** 9)
            if self.encoder_timestamps[j] > self.fog_timestamps[i]:
                omega = self.fog_data[i][-1] / t_fog
                state += self.compute_state(state[2], t_fog, v, omega)
                self.tra[i] = state[:2]
            else:
                j += 1
                t_encoder = (self.encoder_timestamps[j] - self.encoder_timestamps[j - 1]) / (10 ** 9)
                encoder_left_diff = self.encoder_data[j][0] - self.encoder_data[j - 1][0]
                encoder_right_diff = self.encoder_data[j][1] - self.encoder_data[j - 1][1]
                v_left = self.compute_velocity(self.left_diameter, encoder_left_diff, t_encoder)
                v_right = self.compute_velocity(self.right_diameter, encoder_right_diff, t_encoder)
                v = (v_left + v_right) / 2
                state += self.compute_state(state[2], t_fog, v, omega)
                self.tra[i] = state[:2]

    def compute_state(self, theta, t, v, omega):
        return t * np.array([v * np.cos(theta), v * np.sin(theta), omega])

    def compute_velocity(self, diameter, encoder_data_diff, t):
        return (np.pi * diameter * encoder_data_diff) / self.resolution / t

    def plot_trajectory(self):
        self.compute_trajectory()
        temp = self.tra.shape[0] // 6
        for i in range(1, 7):
            plt.plot(self.tra[: i * temp, 0], self.tra[: i * temp, 1])
            plt.title('Motion-Only Localized Map')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig("fig/motion_only_localized_map_{0}.jpg".format(i + 1))
            plt.show(block = True)

if __name__ == "__main__":
    encoder_resolution = 4096
    encoder_left_wheel_diameter = 0.623479
    encoder_right_wheel_diameter = 0.622806
    encoder_wheel_base = 1.52439
    theTrajectory = trajectory(encoder_resolution, encoder_left_wheel_diameter,
                               encoder_right_wheel_diameter, encoder_wheel_base)
    theTrajectory.plot_trajectory()
