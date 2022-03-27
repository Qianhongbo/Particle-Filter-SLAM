# Particle Filter SLAM

## Summary

This project aims to use data captured from a variety of sensors including Encoder, FOG and LiDAR on the vehicle to map and localize its surrounding environment. The method to achieve this is Simultaneous Localization and Mapping (SLAM) with particle filter, which is the most popular method in this topic. This enables the vehicle to to perceive the external environment, which can be helpful for autonomous driving.

## Data

- param.zip: https://drive.google.com/file/d/1alGT4ZJcNKCLLLW-DIbt50p8ZBFrUyFZ/view?usp=sharing

- sensor data.zip: https://drive.google.com/file/d/1s82AV_TACqqgCJaE6B561LL1rgWmewey/view?usp=sharing

- stereo images.zip: https://drive.google.com/file/d/1kJHOm9-1Zz13rUBg46D3RS1TOeXZEhQS/view?usp=sharing

## Requirement

opencv-python == 4.2.0.34

matplotlib >= 2.2

numpy

pandas

scipy

scikit-image >= 0.14.0

## Particle Filter SLAM

To run the code, run the scripts

map.py

trajectory.py

particle_filter.py

## Result

![motion_only_localized_map_7](https://tva1.sinaimg.cn/large/e6c9d24egy1gzve92em10j20hs0dcq3b.jpg)

![test_1150000](https://tva1.sinaimg.cn/large/e6c9d24egy1gzve9yrpiqj20hs0dcjrs.jpg)