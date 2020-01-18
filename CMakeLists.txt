cmake_minimum_required(VERSION 2.8.3)
project(graspit_commander)

find_package(catkin REQUIRED COMPONENTS rospy shape_reconstruction
                                        grid_sample_plugin)

catkin_python_setup()
