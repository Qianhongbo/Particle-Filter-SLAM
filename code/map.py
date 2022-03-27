import numpy as np
from pr2_utils import read_data_from_csv
from pr2_utils import bresenham2D
import matplotlib.pyplot as plt

class Map:
    def __init__(self, res, x_min, x_max, y_min, y_max):
        self.res = res
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        size_x, size_y = int(np.ceil((x_max - x_min) / res + 1)), int(np.ceil((y_max - y_min) / res + 1))
        self.map = np.zeros((size_x, size_y), dtype='float64')
        self.get_lidar_data()

    def get_lidar_data(self):
        """
        read data from "lidar.csv"
        """
        lidar_timestamps, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
        ranges = lidar_data[0, :]
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        validIdx = np.logical_and((ranges < 70), (ranges > 2))
        self.ranges = ranges[validIdx]
        self.angles = angles[validIdx]

    def lidar_to_world(self, position):
        """
        transform lidar scan data to world frame
        """
        lidar_rotation = np.array([[0.00130201, 0.796097, 0.605167],
                                   [0.999999, -0.000419027, -0.00160026],
                                   [-0.00102038, 0.605169, -0.796097]])
        lidar_translation = np.array([0.8349, -0.0126869, 1.76416])
        bTl = self.rp_to_pose(lidar_rotation, lidar_translation)
        wTb = self.theta_to_rotation(position)
        vehicle_pose = np.vstack((self.ranges * np.cos(self.angles), self.ranges * np.sin(self.angles),
                                  np.ones(self.ranges.shape[0]), np.ones(self.ranges.shape[0])))
        return np.dot(wTb, np.dot(bTl, vehicle_pose))[:2, :]

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

    def plot_first_scan(self, position):
        # plot and save the laser data
        pos_world = self.lidar_to_world(position)
        plt.scatter(pos_world[0], pos_world[1])
        plt.title("Laser data")
        plt.savefig("fig/laser_data.jpg")
        plt.show(block=True)
        # plot and save the occupancy grid map
        for i in range(pos_world.shape[1]):
            px, py = 0, 0
            start_point_x = np.ceil((px - self.x_min) / self.res).astype(np.int16) - 1
            start_point_y = np.ceil((py - self.y_min) / self.res).astype(np.int16) - 1
            xis = np.ceil((pos_world[0][i] - self.x_min) / self.res).astype(np.int16) - 1
            yis = np.ceil((pos_world[1][i] - self.y_min) / self.res).astype(np.int16) - 1
            x, y  = bresenham2D(start_point_x, start_point_y, xis, yis)
            self.map[xis][yis] += np.log(4)
            self.map[x[:-1].astype(int), y[:-1].astype(int)] -= np.log(4)
        flip_map = np.flip(self.map.T, axis=0)
        plt.imshow(flip_map, cmap = "binary")
        plt.title("Occupancy grid map")
        plt.savefig("fig/occupancy_grid_map.jpg")
        plt.show(block = True)

if __name__ == "__main__":
    theMap = Map(1, -100, 1500, -1400, 200)
    start_pos = np.array((0, 0, 0))  # start position
    theMap.plot_first_scan(start_pos)
