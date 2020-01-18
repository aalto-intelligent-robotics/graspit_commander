#!/usr/bin/env python

import rospy
from graspit_commander import GraspitCommander
from graspit_interface.msg import PlanXBestGraspsAction, PlanXBestGraspsActionResult
from graspit_interface.msg import SimulationExperimentAction, SimulationExperimentActionResult
import actionlib
import os
from shape_reconstruction.utils import file_utils, shape_completion_utils
import nets as sn
import numpy as np
import grid_sample_client


class GraspitCommanderNode(object):
    def __init__(self):
        self.plan_best_grasps_action_server = actionlib.SimpleActionServer(
            'plan_best_grasp',
            PlanXBestGraspsAction,
            execute_cb=self.plan_and_return_best_x_grasps,
            auto_start=False)
        self.simulation_experiment_action_server = actionlib.SimpleActionServer(
            'simulation_experiment',
            SimulationExperimentAction,
            execute_cb=self.simulation_experiment_callback,
            auto_start=False)

        self.plan_best_grasps_action_server.start()
        self.simulation_experiment_action_server.start()

    def import_robot(self, robot_name, pose=None):
        if pose == None:
            GraspitCommander.importRobot(robot_name)
        else:
            GraspitCommander.importRobot(robot_name, pose)

    def save_grasps(self, grasps, folder, object_name):
        filename = object_name + "_sampled_grasps.txt"
        grasp_file = file_utils.create_file(folder, filename)
        with open(grasp_file, 'w') as f:
            for grasp in grasps:
                pose_as_string = str(grasp.pose.position.x) + " " + str(grasp.pose.position.y) + " "\
                    + str(grasp.pose.position.z) + " " + str(grasp.pose.orientation.x) + " " \
                    + str(grasp.pose.orientation.y) + " " + str(grasp.pose.orientation.z) + " "\
                    + str(grasp.pose.orientation.w) + "\n"
                file_utils.write_to_file(f, pose_as_string)

    def convert_pose_msg_to_str(self, pose):
        pose_string = str(pose.position.x) + " " + str(pose.position.y) + " " \
            + str(pose.position.z) + " " + str(pose.orientation.x) + " " \
            + str(pose.orientation.y) + " " + str(pose.orientation.z) + " "\
            + str(pose.orientation.w) + "\n"
        return pose_string

    def convert_pose_msg_to_list(self, pose):
        pose_as_list = [
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z,
            pose.orientation.w
        ]
        return pose_as_list

    def convert_pose_msg_to_numpy(self, pose):
        return np.array(self.convert_pose_msg_to_list(pose))

    def save_gripper_poses(self, poses, folder, filename):
        pose_file = file_utils.create_file(folder, filename)
        with open(pose_file, 'w') as f:
            for pose in poses:
                pose_string = self.convert_pose_msg_to_str(pose)
                file_utils.write_to_file(f, pose_string)

    def import_grasps_from_folder(self, folder):
        all_files = file_utils.get_all_files_in_folder(folder)
        grasp_file = file_utils.get_objects_matching_pattern(
            all_files, "sampled_grasps")
        grasp_poses = []
        with open(grasp_file[0], 'r') as f:
            for line in f:
                data = line.split(" ")
                grasp_pose = msg.Pose()
                grasp_pose.position.x = float(data[0].split(" ")[-1])
                grasp_pose.position.y = float(data[1].split(" ")[-1])
                grasp_pose.position.z = float(data[2].split(" ")[-1])
                grasp_pose.orientation.x = float(data[3].split(" ")[-1])
                grasp_pose.orientation.y = float(data[4].split(" ")[-1])
                grasp_pose.orientation.z = float(data[5].split(" ")[-1])
                grasp_pose.orientation.w = float(data[6].split(" ")[-1])
                grasp_poses.append(grasp_pose)

        return grasp_poses

    def import_graspable_object(self, object_name, pose=None):
        if pose == None:
            GraspitCommander.importGraspableBody(object_name)
        else:
            GraspitCommander.importGraspableBody(object_name, pose)

    def evaluate_grasps_on_single_object(self,
                                         object_to_grasp,
                                         grasps,
                                         object_pose=None):
        self.clear_world()
        gripper_joint_position_for_each_grasp = []
        gripper_pose_for_each_grasp = []
        epsilon_grasping_quality_for_each_grasp = []
        volume_grasping_quality_for_each_grasp = []
        self.import_graspable_object(object_to_grasp, object_pose)
        robot_start_pose = shape_completion_utils.set_pose_msg([0, 0, -0.1])
        self.import_robot("BarrettBH8_280", robot_start_pose)
        for grasp in grasps:
            grasp_result, grasp_quality = self.evaluate_pregrasp(grasp)
            gripper_joint_position_for_each_grasp.append(
                grasp_result.robot.joints[0].position)
            gripper_pose_for_each_grasp.append(
                self.convert_pose_msg_to_list(grasp_result.robot.pose))
            epsilon_grasping_quality_for_each_grasp.append(
                grasp_quality.epsilon)
            volume_grasping_quality_for_each_grasp.append(grasp_quality.volume)

        return np.array(epsilon_grasping_quality_for_each_grasp), np.array(
            volume_grasping_quality_for_each_grasp), np.array(
                gripper_joint_position_for_each_grasp), np.array(
                    gripper_pose_for_each_grasp)

    def evaluate_grasps_on_multiple_objects(self,
                                            all_objects_to_grasp,
                                            grasps,
                                            object_pose=None):

        epsilon_quality_metric_for_all_objects_and_grasps = []
        volume_quality_metric_for_all_objects_and_grasps = []
        gripper_joint_position_for_all_objects_and_grasps = []
        gripper_pose_for_all_objects_and_grasps = []
        for object_to_grasp in all_objects_to_grasp:
            grasp_properties_for_each_grasp = self.evaluate_grasps_on_single_object(
                object_to_grasp, grasps, object_pose)
            epsilon_quality_metric_for_all_objects_and_grasps.append(
                grasp_properties_for_each_grasp[0])
            volume_quality_metric_for_all_objects_and_grasps.append(
                grasp_properties_for_each_grasp[1])
            gripper_joint_position_for_all_objects_and_grasps.append(
                grasp_properties_for_each_grasp[2])
            gripper_pose_for_all_objects_and_grasps.append(
                grasp_properties_for_each_grasp[3])
        return np.asarray(
            epsilon_quality_metric_for_all_objects_and_grasps
        ).squeeze(), np.asarray(
            volume_quality_metric_for_all_objects_and_grasps
        ).squeeze(
        ), gripper_joint_position_for_all_objects_and_grasps, gripper_pose_for_all_objects_and_grasps

    def rank_grasps(self, epsilon_grasping_quality, volume_grasping_quality):
        return np.argsort(-1 * epsilon_grasping_quality), np.argsort(
            -1 * volume_grasping_quality)

    def evaluate_pregrasp(self, pre_grasp):
        GraspitCommander.toggleAllCollisions(False)
        GraspitCommander.forceRobotDof([pre_grasp.dofs[0], 0, 0, 0])
        GraspitCommander.setRobotPose(pre_grasp.pose)
        GraspitCommander.toggleAllCollisions(True)
        GraspitCommander.findInitialContact()
        GraspitCommander.approachToContact(250)
        GraspitCommander.autoGrasp()
        result = GraspitCommander.getRobot(0)
        quality = self.compute_quality()
        return result, quality

    def compute_quality(self):
        return GraspitCommander.computeQuality()

    def get_quality_from_grasps(self, grasps):
        epsilon_grasping_quality = []
        volume_grasping_quality = []
        for grasp in grasps:
            epsilon_grasping_quality.append(grasp.epsilon_quality)
            volume_grasping_quality.append(grasp.volume_quality)
        return epsilon_grasping_quality, volume_grasping_quality

    def get_all_mesh_samples(self, path_to_folder_with_meshes):
        file_path = ("/").join(
            path_to_folder_with_meshes.split("/")[:-1]) + "/"
        all_files = file_utils.get_all_files_in_folder(file_path)
        samples = file_utils.get_objects_matching_pattern(all_files, "sample")
        sample_xml = file_utils.get_objects_matching_pattern(samples, "xml")
        return sample_xml

    def average_grasp_quality_over_samples(self, quality):
        return quality.mean(axis=0)

    def rank_grasps_on_mesh_samples(self, grasp_results_on_mesh_samples):
        epsilon_quality_metric_of_all_grasps = grasp_results_on_mesh_samples[0]
        volume_quality_metric_of_all_grasps = grasp_results_on_mesh_samples[1]
        epsilon_quality_metric_of_all_grasps[
            epsilon_quality_metric_of_all_grasps == -1] = 0
        average_epsilon_quality_of_all_grasps = self.average_grasp_quality_over_samples(
            epsilon_quality_metric_of_all_grasps)
        average_volume_quality_of_all_grasps = self.average_grasp_quality_over_samples(
            volume_quality_metric_of_all_grasps)
        return self.rank_grasps(average_epsilon_quality_of_all_grasps,
                                average_volume_quality_of_all_grasps)

    def rank_grasps_on_single_mesh(self, grasp_results_on_single_mesh):
        epsilon_quality_metric_of_all_grasps = grasp_results_on_single_mesh[0]
        volume_quality_metric_of_all_grasps = grasp_results_on_single_mesh[1]
        epsilon_quality_metric_of_all_grasps[
            epsilon_quality_metric_of_all_grasps == -1] = 0
        return self.rank_grasps(epsilon_quality_metric_of_all_grasps,
                                volume_quality_metric_of_all_grasps)

    def sample_grasps(self, planner, max_steps=50000):
        if planner.lower() == "simulated_annealing":
            grasps = GraspitCommander.planGrasps(max_steps=max_steps)
        elif planner.lower() == "uniform":
            grasps = grid_sample_client.GridSampleClient.computePreGrasps(5, 2)
        else:
            print("Planner " + planner +
                  " not found.\nDefaulting to the simulated annealing planner")
            grasps = GraspitCommander.planGrasps(max_steps=50000)
        return grasps.grasps

    def clear_world(self):
        try:
            GraspitCommander.clearWorld()
        except:
            print "Could not call graspit commander clearWold. Will try again"
            try:
                GraspitCommander.clearWorld()
            except:
                print "Could not clear world so I will just continue"

    def setup_scene_with_barrett_hand(self, add_collision_plane=False):
        self.clear_world()
        if add_collision_plane:
            GraspitCommander.importObstacle("floor")
        self.import_robot("BarrettBH8_280")

    def get_object_name(self, object_mean_mesh):
        return object_mean_mesh.replace('_mean_shape', '')

    def get_quality_metric_of_ranked_grasp_on_ground_truth_mesh(
            self, grasp_results, grasp_indices):
        epsilon_quality_metric_of_all_grasps = grasp_results[0]
        volume_quality_metric_of_all_grasps = grasp_results[1]
        epsilon_quality_metric_of_all_grasps[
            epsilon_quality_metric_of_all_grasps == -1] = 0
        best_epsilon_grasp_indices, best_volume_grasp_indices = grasp_indices
        return epsilon_quality_metric_of_all_grasps[best_epsilon_grasp_indices[
            0]], volume_quality_metric_of_all_grasps[
                best_volume_grasp_indices[0]]

    def do_simulation_experiment(self,
                                 shape_completed_meshes,
                                 ground_truth_meshes,
                                 evaluate_on_samples,
                                 folder_to_save_results,
                                 planner="uniform"):
        simulation_experiment_results = {}
        for test_case, shape_completed_mesh_files in shape_completed_meshes.items(
        ):
            print("Simulating grasps for objects in " + test_case)
            mean_meshes = shape_completed_mesh_files[0]
            if evaluate_on_samples:
                sample_meshes = shape_completed_mesh_files[1]

            results_per_object = {}
            for mean_mesh in mean_meshes:
                results_per_method = {}
                object_name = self.get_object_name(
                    file_utils.strip_file_definition(mean_mesh))
                self.setup_scene_with_barrett_hand()
                self.import_graspable_object(mean_mesh)
                grasps = self.sample_grasps(planner)
                if not grasps:
                    print("No grasps found for object " + mean_mesh +
                          ". Continuing to next object")
                    continue
                self.save_grasps(
                    grasps,
                    folder_to_save_results + "/" + test_case + "/grasps/",
                    object_name)

                ground_truth_mesh = shape_completion_utils.get_ground_truth_mesh_corresponding_to_shape_completed_mesh(
                    ground_truth_meshes, mean_mesh)

                grasp_results_on_ground_truth_mesh = self.evaluate_grasps_on_single_object(
                    ground_truth_mesh, grasps)

                self.save_grasp_results(
                    grasp_results_on_ground_truth_mesh,
                    folder_to_save_results + "/" + test_case +
                    "/grasp_measures/ground_truth_mesh/", object_name)

                grasp_results_on_mean_mesh = self.evaluate_grasps_on_single_object(
                    mean_mesh, grasps)
                indices_of_best_grasps = self.rank_grasps_on_single_mesh(
                    grasp_results_on_mean_mesh)
                quality_metric_on_ground_truth_of_best_mean_mesh_grasp = self.get_quality_metric_of_ranked_grasp_on_ground_truth_mesh(
                    grasp_results_on_ground_truth_mesh, indices_of_best_grasps)
                self.save_grasp_results(
                    grasp_results_on_mean_mesh, folder_to_save_results + "/" +
                    test_case + "/grasp_measures/mean_mesh/", object_name)
                results_per_method[
                    "mean mesh"] = quality_metric_on_ground_truth_of_best_mean_mesh_grasp

                if evaluate_on_samples:
                    mesh_samples = file_utils.get_objects_matching_pattern(
                        sample_meshes, object_name)
                    grasp_results_on_mesh_samples = self.evaluate_grasps_on_multiple_objects(
                        mesh_samples, grasps)
                    indices_of_best_grasps = self.rank_grasps_on_mesh_samples(
                        grasp_results_on_mesh_samples)
                    quality_metric_on_ground_truth_of_best_mesh_sample_grasps = self.get_quality_metric_of_ranked_grasp_on_ground_truth_mesh(
                        grasp_results_on_ground_truth_mesh,
                        indices_of_best_grasps)
                    results_per_method[
                        "sampled meshes"] = quality_metric_on_ground_truth_of_best_mesh_sample_grasps
                    self.save_grasp_results_on_samples(
                        grasp_results_on_mesh_samples, folder_to_save_results +
                        "/" + test_case + "/grasp_measures/mesh_samples/",
                        object_name)

                results_per_object[object_name] = results_per_method
            simulation_experiment_results[test_case] = results_per_object

        return simulation_experiment_results

    def save_quality_metrics(self, quality_metric, folder, filename):
        np.savetxt(file_utils.create_file(folder, filename),
                   quality_metric,
                   delimiter="\n")

    def save_grasp_results_on_samples(self, grasp_results, folder,
                                      object_name):
        epsilon_quality_for_each_grasp, volume_quality_for_each_grasp, gripper_joint_position_for_each_grasp, gripper_pose_for_each_grasp = grasp_results
        average_epsilon_quality_of_all_grasps = self.average_grasp_quality_over_samples(
            epsilon_quality_for_each_grasp)
        average_volume_quality_of_all_grasps = self.average_grasp_quality_over_samples(
            volume_quality_for_each_grasp)
        self.save_quality_metrics(average_epsilon_quality_of_all_grasps,
                                  folder, object_name + "_epsilon_quality.txt")
        self.save_quality_metrics(average_volume_quality_of_all_grasps, folder,
                                  object_name + "_volume_quality.txt")

    def save_grasp_results(self, grasp_results, folder, object_name):
        epsilon_quality_for_each_grasp, volume_quality_for_each_grasp, gripper_joint_position_for_each_grasp, gripper_pose_for_each_grasp = grasp_results
        self.save_quality_metrics(epsilon_quality_for_each_grasp, folder,
                                  object_name + "_epsilon_quality.txt")
        self.save_quality_metrics(volume_quality_for_each_grasp, folder,
                                  object_name + "_volume_quality.txt")
        np.savetxt(
            file_utils.create_file(folder, object_name + "_gripper_joint.txt"),
            gripper_joint_position_for_each_grasp)
        np.savetxt(file_utils.create_file(folder,
                                          object_name + "_gripper_pose.txt"),
                   gripper_pose_for_each_grasp,
                   header="x, y, z, i, j, k, w")

    def add_tuples_together(self, tuple_a, tuple_b):
        return tuple(map(sum, zip(tuple_a, tuple_b)))

    def analyze_simulation_experiment_results(self,
                                              simulation_experiment_results):
        average_grasp_quality_metric_per_test_case_per_method = {}
        for test_case, results_per_object in simulation_experiment_results.items(
        ):
            average_quality_metric_per_method = {}
            number_of_objects = 0
            for object_name, results_per_method_per_object in results_per_object.items(
            ):
                number_of_objects += 1
                for method_name, results_for_each_method in results_per_method_per_object.items(
                ):
                    # Check if method is not in dictionary
                    if method_name not in average_quality_metric_per_method:
                        average_quality_metric_per_method[
                            method_name] = results_for_each_method
                    else:
                        average_quality_metric_per_method[
                            method_name] = self.add_tuples_together(
                                average_quality_metric_per_method[method_name],
                                results_for_each_method)

            for method_name, summed_quality_metric_for_all_objects in average_quality_metric_per_method.items(
            ):
                average_quality_metric_per_method[method_name] = map(
                    lambda x: x / number_of_objects,
                    summed_quality_metric_for_all_objects)

            average_grasp_quality_metric_per_test_case_per_method[
                test_case] = average_quality_metric_per_method

        return average_grasp_quality_metric_per_test_case_per_method

    def save_simulation_results(self, simulation_results,
                                folder_to_save_results):
        for test_case, results_per_test_case in simulation_results.items():
            filename = file_utils.create_file(
                folder_to_save_results + "/" + test_case + "/",
                "simulation_results.csv")
            with open(filename, 'w') as f:
                data = test_case + ", Epsilon quality metric, Volume Quality Metric\n"
                for method, results_per_method in results_per_test_case.items(
                ):
                    data += method + "," + str(
                        results_per_method[0]) + ',' + str(
                            results_per_method[1]) + "\n"
                file_utils.write_to_file(f, data)

    def simulation_experiment_callback(self, goal):
        print("Doing the simulation experiment")
        ground_truth_meshes, shape_completed_meshes = shape_completion_utils.load_ground_truth_and_shape_completed_meshes(
            goal.path_to_ground_truth_meshes,
            goal.path_to_shape_completed_meshes,
            goal.evaluate_on_shape_samples)
        simulation_experiment_results = self.do_simulation_experiment(
            shape_completed_meshes,
            ground_truth_meshes,
            goal.evaluate_on_shape_samples,
            goal.path_to_save_location,
            planner="uniform")
        simulation_results = self.analyze_simulation_experiment_results(
            simulation_experiment_results)
        self.save_simulation_results(simulation_results,
                                     goal.path_to_save_location)
        result = SimulationExperimentActionResult()
        result.result.success = True

        self.simulation_experiment_action_server.set_succeeded(result.result)

    def plan_and_return_best_x_grasps(self, goal):
        print("Generating best grasps")
        print("Filename " + goal.filename)
        self.setup_scene_with_barrett_hand(add_collision_plane=True)
        mean_mesh_file_name, path_to_graspit_meshes = file_utils.get_file_name_and_path_from_folder_path(
            goal.filename)
        self.import_graspable_object(mean_mesh_file_name)
        grasps = self.sample_grasps(goal.planner)
        number_of_top_x_grasps_to_return = goal.number_of_top_x_grasps_to_return
        if goal.rank_grasps_on_samples:
            print("Ranking grasps on samples")
            sample_xml = self.get_all_mesh_samples(path_to_graspit_meshes)
            grasp_results_on_mesh_samples = self.evaluate_grasps_on_multiple_objects(
                sample_xml, grasps)

            grasps_ranked_according_to_epsilon_metric, grasps_ranked_according_to_volume_metric = self.rank_grasps_on_mesh_samples(
                grasp_results_on_mesh_samples)
        else:
            grasp_results_on_mean_mesh = self.evaluate_grasps_on_single_object(
                mean_mesh_file_name, grasps)
            grasps_ranked_according_to_epsilon_metric, grasps_ranked_according_to_volume_metric = self.rank_grasps_on_single_mesh(
                grasp_results_on_mean_mesh)

        result = PlanXBestGraspsActionResult()
        result.result.top_x_epsilon_quality_grasps = grasps_ranked_according_to_epsilon_metric[:
                                                                                               number_of_top_x_grasps_to_return]
        result.result.top_x_volume_quality_grasps = grasps_ranked_according_to_epsilon_metric[:
                                                                                              number_of_top_x_grasps_to_return]
        for grasp in grasps.grasp:
            result.result.hand_joint_states.append(grasp.dofs)
            result.result.hand_poses.append(grasp.pose)

        result.result.success = True

        self.plan_best_grasps_action_server.set_succeeded(result.result)


if __name__ == '__main__':
    rospy.init_node('planBestGraspNode')
    server = GraspitCommanderNode()
    rospy.spin()
