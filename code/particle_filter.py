import numpy as np
from pr2_utils import read_data_from_csv
from pr2_utils import bresenham2D, mapCorrelation
import matplotlib.pyplot as plt
import random

class Particle_Filter():
    def __init__(self, res, x_min, x_max, y_min, y_max, particle_num, sigma):
        self.res = res
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        size_x, size_y = int(np.ceil((x_max - x_min) / res + 1)), int(np.ceil((y_max - y_min) / res + 1))
        self.map = np.zeros([size_x, size_y], dtype='float64')
        self.load_data()
        self.tra = np.empty([self.encoder_data.shape[0] * 10, 2])
        self.particle_num = particle_num
        self.sigma = sigma
        self.pose_particles = np.zeros([3, self.particle_num])
        self.weights = np.ones(self.particle_num) / self.particle_num

    def load_vehicle_parameter(self, resolution, left_wheel_diameter, right_wheel_diameter):
        self.resolution = resolution
        self.left_diameter = left_wheel_diameter
        self.right_diameter = right_wheel_diameter

    def load_data(self):
        """
        read data from "lidar.csv", "fog.csv" and "lidar.csv"
        """
        self.encoder_timestamps, self.encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')
        self.fog_timestamps, self.fog_data = read_data_from_csv('data/sensor_data/fog.csv')
        self.lidar_timestamps, self.lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')

    def generate_noise(self):
        v_noise = np.random.normal(0, self.sigma, self.particle_num)
        omega_noise = np.random.normal(0, self.sigma, self.particle_num)
        return v_noise, omega_noise

    def occupancy_map(self):
        v, omega, j = 0, 0, 1
        print("starting...")
        for i in range(1, self.fog_timestamps.shape[0]):
            v_noise, omega_noise = self.generate_noise()
            if i % 1000 == 0:
                print(i)
                # save the image
                self.save_img(i)
            t_fog = (self.fog_timestamps[i] - self.fog_timestamps[i - 1]) / (10 ** 9)
            if self.encoder_timestamps[j] > self.fog_timestamps[i]: # sync
                # predict using fog data
                omega = self.fog_data[i][-1] / t_fog + omega_noise
                self.pose_particles += self.compute_state(self.pose_particles[2], t_fog, v + v_noise, omega)
            else:
                # predict using encoder data
                j += 1
                t_encoder = (self.encoder_timestamps[j] - self.encoder_timestamps[j - 1]) / (10 ** 9)
                encoder_left_diff = self.encoder_data[j][0] - self.encoder_data[j - 1][0]
                encoder_right_diff = self.encoder_data[j][1] - self.encoder_data[j - 1][1]
                v_left = self.compute_velocity(self.left_diameter, encoder_left_diff, t_encoder)
                v_right = self.compute_velocity(self.right_diameter, encoder_right_diff, t_encoder)
                v = (v_left + v_right) / 2 + v_noise
                self.pose_particles += self.compute_state(self.pose_particles[2], t_fog, v, omega)
                # update the map
                max_idx = self.correlation(j)
                pose_highest_weight = self.pose_particles[:, max_idx]
                self.tra[j] = self.pose_particles[:, max_idx][:2]
                self.update_map(pose_highest_weight, j, max_idx)

    def save_img(self, idx):
        flip_map = np.flip(self.map.T, axis = 0)
        plt.imshow(flip_map, cmap='binary')
        plt.savefig("fig/another_test_{0}.jpg".format(idx))

    def compute_state(self, theta, t, v, omega):
        return t * np.array([v * np.cos(theta), v * np.sin(theta), omega])

    def compute_velocity(self, diameter, encoder_data_diff, t):
        return (np.pi * diameter * encoder_data_diff) / self.resolution / t

    def correlation(self, j):
        x_im = np.arange(self.x_min, self.x_max + self.res, self.res)
        y_im = np.arange(self.y_min, self.y_max + self.res, self.res)
        x_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
        y_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
        correlation = np.zeros((self.pose_particles.shape[1]))
        for particle in range(self.pose_particles.shape[1]):
            pos_world = self.lidar_to_world(self.pose_particles[:,particle], j)
            # calculate map_correlation
            map_correlation = mapCorrelation(self.map, x_im, y_im, pos_world.T, x_range, y_range)
            # find index of highest value correlation
            correlation[particle] = np.max(map_correlation)
        soft_norm = self.soft_max(correlation)
        self.weights = soft_norm * self.weights / np.sum(soft_norm * self.weights)
        return np.argmax(self.weights)


    def soft_max(self, x):
        x_exp = np.exp(x)
        return x_exp / x_exp.sum()

    def update_map(self, aPose, anIndex, max_idx):
        pose_highest_weight = self.lidar_to_world(aPose, anIndex)
        for j in range(0, pose_highest_weight.shape[0]):
            # start point
            px, py = self.pose_particles[0, max_idx], self.pose_particles[1, max_idx]
            start_point_x = np.ceil((px - self.x_min) / self.res).astype(np.int16) - 1
            start_point_y = np.ceil((py- self.y_min) / self.res).astype(np.int16) - 1
            # end point
            ex, ey = pose_highest_weight[j][0], pose_highest_weight[j][1]
            end_point_x = np.ceil((ex - self.x_min) / self.res).astype(np.int16) - 1
            end_point_y = np.ceil((ey - self.y_min) / self.res).astype(np.int16) - 1
            # use the function in pr2_utlis.py
            x, y = bresenham2D(start_point_x, start_point_y, end_point_x, end_point_y)
            # log_odds
            self.map[x[:-1].astype(int), y[:-1].astype(int)] += np.log(1 / 4)
            for k in range(0, x.shape[0] - 1):
                if self.map[int(x[k]), int(y[k])] < 0:
                    self.map[int(x[k]), int(y[k])] = -1

    def lidar_to_world(self, position, idx):
        """
        transform lidar scan data to world frame
        """
        lidar_rotation = np.array([[0.00130201, 0.796097, 0.605167],
                                   [0.999999, -0.000419027, -0.00160026],
                                   [-0.00102038, 0.605169, -0.796097]])
        lidar_translation = np.array([0.8349, -0.0126869, 1.76416])
        bTl = self.rp_to_pose(lidar_rotation, lidar_translation)
        wTb = self.theta_to_rotation(position)
        xs0, ys0, ones = self.lidar_to_coordinates(idx)
        vehicle_pose = np.vstack((xs0, ys0, ones, ones))
        return np.dot(wTb, np.dot(bTl, vehicle_pose))[:2, :].T

    def rp_to_pose(self, R, p):
        T = np.zeros([4, 4])
        T[:3, :3] = R
        T[:3, -1] = p
        T[-1, -1] = 1
        return T

    def theta_to_rotation(self, position):
        theta = position[2]
        return np.array([[np.cos(theta), -np.sin(theta), 0, position[0]],
                         [np.sin(theta), np.cos(theta), 0, position[1]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def lidar_to_coordinates(self, idx):
        ranges = self.lidar_data[idx, :]
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        validIdx = np.logical_and((ranges < 70), (ranges > 2))
        ranges = ranges[validIdx]
        angles = angles[validIdx]
        return ranges * np.cos(angles), ranges * np.sin(angles), np.ones(ranges.shape[0])


if __name__ == "__main__":
    pf = Particle_Filter(1, -100, 1500, -1400, 200, 30, 0.01)
    encoder_resolution = 4096
    encoder_left_wheel_diameter = 0.623479
    encoder_right_wheel_diameter = 0.622806
    pf.load_vehicle_parameter(encoder_resolution, encoder_left_wheel_diameter, encoder_left_wheel_diameter)
    pf.occupancy_map()