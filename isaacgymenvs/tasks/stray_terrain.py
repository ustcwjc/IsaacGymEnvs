# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os, time
import csv

from isaacgym import gymtorch
from isaacgym import gymapi
from .base.vec_task import VecTask

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse
from isaacgymenvs.tasks.base.vec_task import VecTask


class StrayTerrain(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        if(os.path.exists('data/temp.csv')):
            os.remove('data/temp.csv')

        self.cfg = cfg
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False


        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale_thigh = self.cfg["env"]["control"]["actionScale_thigh"]
        self.action_scale_knee = self.cfg["env"]["control"]["actionScale_knee"]
        self.action_scale_hip = self.cfg["env"]["control"]["actionScale_hip"]
        self.action_scale_spine = self.cfg["env"]["control"]["actionScale_spine"]
        

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"] 
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"] 
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["calfCollisionRewardScale"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["feetStumbleRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["gait_pattern"] = self.cfg["env"]["learn"]["gaitpatternRewardScale"]
        self.rew_scales["power"] = self.cfg["env"]["learn"]["powerRewardScale"]
        self.rew_scales["spine"] = self.cfg["env"]["learn"]["spineRewardScale"]
        self.rew_scales["allfeetonair"] = self.cfg["env"]["learn"]["allfeetonairRewardScale"]
        self.rew_scales["thigh"] = self.cfg["env"]["learn"]["thighRewardScale"]
        self.rew_scales["knee"] = self.cfg["env"]["learn"]["kneeRewardScale"]

        #command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.turning_time = int(self.cfg["env"]["learn"]["turning_s"] / self.dt + 0.5)
        self.allow_calf_contacts = self.cfg["env"]["learn"]["allowcalfContacts"]
        self.Kp_leg = self.cfg["env"]["control"]["stiffness_leg"]
        self.Kd_leg = self.cfg["env"]["control"]["damping_leg"]
        self.Kp_spine = self.cfg["env"]["control"]["stiffness_spine"]
        self.Kd_spine = self.cfg["env"]["control"]["damping_spine"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.base_height_des = self.cfg["env"]["baseInitState"]["base_height_des"]
        self.feet_air_time_des = self.cfg["env"]["control"]["feetAirTimeDesired"]
        self.gait_pattern_command = self.cfg["env"]["control"]["gaitpattern"]
        self.command_adjustment_rate = int(self.cfg["env"]["control"]["command_adjustment_rate"]/self.dt)


        

        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.dt = self.decimation * self.dt
        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        


        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        # self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        # self.rigid_body_state = self.rigid_body_state.view(self.num_envs, 21, 13)
        # print("##################################",self.rigid_body_state.shape)

        # initialize some data used later on
        self.common_step_counter = 0
        self.individual_step_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_stance_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time_LF = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time_RF = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time_LH = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time_RH = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # self.foot_vel_sensor = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.timer = 0
        self.gait_pattern = torch.zeros(self.num_envs, 1, dtype=torch.int, device=self.device, requires_grad=False)
        self.phase_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.commands_buf = torch.zeros(self.num_envs, 3, self.command_adjustment_rate, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel
        self.command_counter = 0

        self.height_points = self.init_height_points()
        self.measured_heights = 0
        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(), "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "gait_pattern": torch_zeros(), "power": torch_zeros(),  "collision": torch_zeros(),
                               "stumble": torch_zeros(), "action_rate": torch_zeros(), "spine": torch_zeros(), "hip": torch_zeros(), 
                               "allfeetonair": torch_zeros(), "thigh": torch_zeros(), "knee": torch_zeros()}

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self.cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:176] = self.cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        noise_vec[176:188] = 0. # previous actions
        return noise_vec

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        stray_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(stray_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(stray_asset)

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(stray_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)

        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(stray_asset)
        self.dof_names = self.gym.get_asset_dof_names(stray_asset)
        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        calf_name = self.cfg["env"]["urdfAsset"]["calfName"]
        thigh_name = self.cfg["env"]["urdfAsset"]["thighName"]
        feet_names = [s for s in body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        calf_names = [s for s in body_names if calf_name in s]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        thigh_names = [s for s in body_names if thigh_name in s]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(stray_asset)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.stray_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)
            
            

            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(stray_asset, rigid_shape_prop)
            stray_handle = self.gym.create_actor(env_handle, stray_asset, start_pose, "stray", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, stray_handle, dof_props)
            self.envs.append(env_handle)
            self.stray_handles.append(stray_handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_handle, stray_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.3,0.3,0.3))

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.stray_handles[0], feet_names[i])
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.stray_handles[0], calf_names[i])
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.stray_handles[0], thigh_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.stray_handles[0], "base")
        self.hind_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.stray_handles[0], "base_hind")

        # self.foot_vel_sensor = self.gym.create_actor_sensor(self.envs, self.feet_indices, gymapi.SENSOR_TYPE_RIGID_LIN_VEL)

    def check_termination(self):
        self.reset_buf = torch.norm(self.projected_gravity[..., 0:2], dim=1) > 0.99
        self.reset_buf |= torch.norm(self.contact_forces[..., self.base_index, :], dim=1) > 1
        self.reset_buf |= torch.norm(self.contact_forces[..., self.hind_body_index, :], dim=1) > 1
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.reset_buf |= torch.norm(base_height, dim=0) < 0.15
        if not self.allow_calf_contacts:
            calf_contact = torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1.
            self.reset_buf |= torch.any(calf_contact, dim=1)
        # thigh_contact = torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1.
        # self.reset_buf |= torch.any(thigh_contact, dim=1)

        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        self.measured_heights = self.get_heights()
        # heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale
        self.obs_buf = torch.cat((  self.base_lin_vel * self.lin_vel_scale,
                                    self.base_ang_vel  * self.ang_vel_scale,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale * torch.clip(self.individual_step_counter/100,0,1),
                                    self.dof_pos * self.dof_pos_scale,
                                    self.dof_vel * self.dof_vel_scale,
                                    self.actions,
                                    (self.phase_time-self.dt)*5,
                                    ), dim=-1)

    def compute_reward(self):

        #priority conditions
        # base_acc = 
        # stable_condition = 

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"]

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # base height penalty
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        rew_base_height = torch.square(base_height - self.base_height_des) * self.rew_scales["base_height"] 

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        # collision penalty
        calf_contact = torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1.
        rew_collision = torch.sum(calf_contact, dim=1) * self.rew_scales["collision"] # sum vs any ?
        
        # spine reward so that spine dont move too much
        rew_spine = torch.sum(torch.square(self.dof_pos[:, 6:8] - self.default_dof_pos[:, 6:8]), dim=1) * self.rew_scales["spine"]

        # stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        # power penalty
        power = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        rew_power = power * self.rew_scales["power"]
        # air time reward
        # contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        first_flight = (self.feet_air_time == 0.) * (~contact)
        self.feet_air_time += self.dt
        self.feet_stance_time += self.dt
        rew_airTime = torch.sum(torch.abs(self.feet_air_time - self.feet_air_time_des*0.5) * first_contact, dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        # rew_airTime += torch.sum((self.feet_stance_time - self.feet_air_time_des) * first_flight, dim=1) * self.rew_scales["air_time"] # reward only on first flight off the ground
        # rew_airTime = torch.sum(
        #     torch.abs(
        #         torch.abs(torch.abs(self.feet_air_time[...,(0,3)]) - 
        #     torch.abs(self.feet_air_time[...,(1,2)])) -
        #     self.feet_air_time_des), dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground

        
        if self.timer > self.feet_air_time_des:
            self.timer=0
            self.phase_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        



        #gait pattern reward
            
        # for i in range(self.num_envs):
        #     self.gait_pattern[i] = self.gait_pattern_command

        
        # vel_foot_LF = self.rigid_body_state[..., self.feet_indices[0], 7:10]
        # vel_foot_RF = self.rigid_body_state[..., self.feet_indices[1], 7:10]
        # vel_foot_LH = self.rigid_body_state[..., self.feet_indices[2], 7:10]
        # vel_foot_RH = self.rigid_body_state[..., self.feet_indices[3], 7:10]
        # quat_foot_LF = self.rigid_body_state[..., self.feet_indices[0], 3:7]
        # quat_foot_RF = self.rigid_body_state[..., self.feet_indices[1], 3:7]
        # quat_foot_LH = self.rigid_body_state[..., self.feet_indices[2], 3:7]
        # quat_foot_RH = self.rigid_body_state[..., self.feet_indices[3], 3:7]
        # vel_foot_LF = quat_rotate_inverse(quat_foot_LF, vel_foot_LF)
        # vel_foot_RF = quat_rotate_inverse(quat_foot_RF, vel_foot_RF)
        # vel_foot_LH = quat_rotate_inverse(quat_foot_LH, vel_foot_LH)
        # vel_foot_RH = quat_rotate_inverse(quat_foot_RH, vel_foot_RH)
        # print("vel_foot_LF",vel_foot_LF[0,...])
        # print("vel_foot_RF",vel_foot_RF)
        # print("vel_foot_LH",vel_foot_LH)
        # print("vel_foot_RH",vel_foot_RH[0,...])

        phaser = np.sin(2*np.pi*self.timer/self.feet_air_time_des)
        if self.timer < self.feet_air_time_des*0.5:
            c1=0
            c2=1
        else:
            c1=1
            c2=0
            # F_LF=c1 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[0]]), dim=1) + (1-c1) * torch.sum(torch.square(0.1*vel_foot_LF), dim=1)
            # F_RF=c2 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[1]]), dim=1) + (1-c2) * torch.sum(torch.square(0.1*vel_foot_RF), dim=1)
            # F_LH=c2 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[2]]), dim=1) + (1-c2) * torch.sum(torch.square(0.1*vel_foot_LH), dim=1)
            # F_RH=c1 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[3]]), dim=1) + (1-c1) * torch.sum(torch.square(0.1*vel_foot_RH), dim=1)
            # F_LF *= phaser*phaser
            # F_RF *= phaser*phaser
            # F_LH *= phaser*phaser
            # F_RH *= phaser*phaser
        F_LF=c1*contact[...,0] + (1-c1)*~contact[...,0]
        F_RF=c2*contact[...,1] + (1-c2)*~contact[...,1]
        F_LH=c2*contact[...,2] + (1-c2)*~contact[...,2]
        F_RH=c1*contact[...,3] + (1-c1)*~contact[...,3]
            # gait_pattern = (torch.exp(-F_LF) + torch.exp(-F_RF) + torch.exp(-F_LH) + torch.exp(-F_RH))/4
        gait_pattern_trot = torch.exp(-(F_LF+F_RF+F_LH+F_RH))
            # gait_pattern = ~contact[...,0]*contact[...,1]*contact[...,2]*~contact[...,3] + contact[...,0]*~contact[...,1]*~contact[...,2]*contact[...,3]

        phaser = np.sin(2*np.pi*self.timer/self.feet_air_time_des)
        if self.timer < self.feet_air_time_des*0.5:
            c1=0
            c2=1
        else:
            c1=1
            c2=0
            # F_LF=c1 * torch.square(0.05*self.contact_forces[:, self.feet_indices[0], 2]) + (1-c1) * torch.sum(torch.square(0.2*vel_foot_LF), dim=1)
            # F_RF=c1 * torch.square(0.05*self.contact_forces[:, self.feet_indices[1], 2]) + (1-c1) * torch.sum(torch.square(0.2*vel_foot_RF), dim=1)
            # F_LH=c2 * torch.square(0.05*self.contact_forces[:, self.feet_indices[2], 2]) + (1-c2) * torch.sum(torch.square(0.2*vel_foot_LH), dim=1)
            # F_RH=c2 * torch.square(0.05*self.contact_forces[:, self.feet_indices[3], 2]) + (1-c2) * torch.sum(torch.square(0.2*vel_foot_RH), dim=1)
            # F_LF *= phaser*phaser
            # F_RF *= phaser*phaser
            # F_LH *= phaser*phaser
            # F_RH *= phaser*phaser
        F_LF=c1*contact[...,0] + (1-c1)*~contact[...,0]
        F_RF=c1*contact[...,1] + (1-c1)*~contact[...,1]
        F_LH=c2*contact[...,2] + (1-c2)*~contact[...,2]
        F_RH=c2*contact[...,3] + (1-c2)*~contact[...,3]
        gait_pattern_bound = torch.exp(-(F_LF+F_RF+F_LH+F_RH))
            # gait_pattern = ~contact[...,0]*~contact[...,1]*contact[...,2]*contact[...,3] + contact[...,0]*contact[...,1]*~contact[...,2]*~contact[...,3]

        phaser = np.sin(2*np.pi*self.timer/self.feet_air_time_des)
        if self.timer < self.feet_air_time_des*0.5:
            c1=0
            c2=1
        else:
            c1=1
            c2=0
            # F_LF=c1 * torch.square(0.05*self.contact_forces[:, self.feet_indices[0], 2]) + (1-c1) * torch.sum(torch.square(0.2*vel_foot_LF), dim=1)
            # F_RF=c2 * torch.square(0.05*self.contact_forces[:, self.feet_indices[1], 2]) + (1-c2) * torch.sum(torch.square(0.2*vel_foot_RF), dim=1)
            # F_LH=c1 * torch.square(0.05*self.contact_forces[:, self.feet_indices[2], 2]) + (1-c1) * torch.sum(torch.square(0.2*vel_foot_LH), dim=1)
            # F_RH=c2 * torch.square(0.05*self.contact_forces[:, self.feet_indices[3], 2]) + (1-c2) * torch.sum(torch.square(0.2*vel_foot_RH), dim=1)
            # F_LF *= phaser*phaser
            # F_RF *= phaser*phaser
            # F_LH *= phaser*phaser
            # F_RH *= phaser*phaser
        F_LF=c1*contact[...,0] + (1-c1)*~contact[...,0]
        F_RF=c2*contact[...,1] + (1-c2)*~contact[...,1]
        F_LH=c1*contact[...,2] + (1-c1)*~contact[...,2]
        F_RH=c2*contact[...,3] + (1-c2)*~contact[...,3]
        gait_pattern_pace = torch.exp(-(F_LF+F_RF+F_LH+F_RH))
            # gait_pattern = contact[...,0]*~contact[...,1]*contact[...,2]*~contact[...,3] + ~contact[...,0]*contact[...,1]*~contact[...,2]*contact[...,3]

        if self.timer < self.feet_air_time_des*3/8:
            c_front=0
        else:
            c_front=1
        if self.timer < self.feet_air_time_des*7/8:
            if self.timer >= self.feet_air_time_des*(1/2):
                c_hind=0
            else:
                c_hind=1
        else:
            c_hind=1
        # F_LF=c_LF*contact[...,0] + (1-c_LF)*~contact[...,0]
        # F_RF=c_RF*contact[...,1] + (1-c_RF)*~contact[...,1]
        # F_LH=c_LH*contact[...,2] + (1-c_LH)*~contact[...,2]
        # F_RH=c_RH*contact[...,3] + (1-c_RH)*~contact[...,3]
        F_front = c_front*contact[...,0] + (1-c_front)*~contact[...,0]*~contact[...,1] + c_front*contact[...,1]
        F_hind = c_hind*contact[...,2] + (1-c_hind)*~contact[...,2]*~contact[...,3] + c_hind*contact[...,3]
        gait_pattern_gallop = torch.exp(-2*(F_front+F_hind))
            # contact_front = ~(~contact[...,0]*~contact[...,1])
            # contact_hind = ~(~contact[...,2]*~contact[...,3])
            # no_contact = ~contact[...,0]*~contact[...,1]*~contact[...,2]*~contact[...,3]
            # gait_pattern = contact_front * ~contact_hind * ~no_contact + ~contact_front * contact_hind * ~no_contact + ~contact_front * ~contact_hind * no_contact
        if self.timer < self.feet_air_time_des*3/8:
            c1=0
        else:
            c1=1
        if self.timer < self.feet_air_time_des*7/8:
            if self.timer >= self.feet_air_time_des*4/8:
                c2=0
            else:
                c2=1
        else:
            c2=1
            # F_LF=c1 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[0]]), dim=1) + (1-c1) * torch.sum(torch.square(0.1*vel_foot_LF), dim=1)
            # F_RF=c2 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[1]]), dim=1) + (1-c2) * torch.sum(torch.square(0.1*vel_foot_RF), dim=1)
            # F_LH=c2 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[2]]), dim=1) + (1-c2) * torch.sum(torch.square(0.1*vel_foot_LH), dim=1)
            # F_RH=c1 * torch.sum(torch.square(0.02*self.contact_forces[:, self.feet_indices[3]]), dim=1) + (1-c1) * torch.sum(torch.square(0.1*vel_foot_RH), dim=1)
            # F_LF *= phaser*phaser
            # F_RF *= phaser*phaser
            # F_LH *= phaser*phaser
            # F_RH *= phaser*phaser
        F_LF=c1*contact[...,0] + (1-c1)*~contact[...,0]
        F_RF=c2*contact[...,1] + (1-c2)*~contact[...,1]
        F_LH=c2*contact[...,2] + (1-c2)*~contact[...,2]
        F_RH=c1*contact[...,3] + (1-c1)*~contact[...,3]
        gait_pattern_flyingtrot = torch.exp(-(F_LF+F_RF+F_LH+F_RH))
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.25 #no reward for zero command
        rew_gait_pattern = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.gait_pattern_command == 0:
            rew_gait_pattern=gait_pattern_trot
        if self.gait_pattern_command == 1:
            rew_gait_pattern=gait_pattern_bound
        if self.gait_pattern_command == 2:
            rew_gait_pattern=gait_pattern_pace
        if self.gait_pattern_command == 3:
            rew_gait_pattern=gait_pattern_gallop
        if self.gait_pattern_command == 4:
            rew_gait_pattern=gait_pattern_flyingtrot

        rew_gait_pattern *= self.rew_scales["gait_pattern"]
        rew_gait_pattern *= torch.norm(self.commands[:, :3], dim=1) > 0.25 #no reward for zero command
        self.feet_air_time *= ~contact
        self.feet_stance_time *= contact
        self.timer += self.dt
        self.phase_time += self.dt

        # print(rew_gait_pattern)


        # print("power",rew_power)
        # print("torque",rew_torque)
        # all feet on air penalty
        all_no_contact = ~contact[:, 0] * ~contact[:, 1] * ~contact[:, 2] * ~contact[:, 3]
        rew_allfeetonair = all_no_contact * self.rew_scales["allfeetonair"]

        # cosmetic penalty for hip motion
        rew_hip = torch.sum(torch.square(self.dof_pos[:, [0, 3, 8, 11]] - self.default_dof_pos[:, [0, 3, 8, 11]]), dim=1)* self.rew_scales["hip"]

        # thigh penalty
        rew_thigh = torch.sum(torch.square(self.dof_pos[:, [1, 4, 9, 12]] - self.default_dof_pos[:, [1, 4, 9, 12]]), dim=1)* self.rew_scales["thigh"]

        # knee penalty
        rew_knee = torch.sum(torch.square(self.dof_pos[:, [2, 5, 10, 13]] - self.default_dof_pos[:, [2, 5, 10, 13]]), dim=1)* self.rew_scales["knee"]

        # total reward
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + rew_thigh +\
                    rew_torque + rew_joint_acc + rew_collision + rew_action_rate + rew_airTime + rew_hip + rew_stumble + rew_gait_pattern + rew_spine + rew_power + rew_allfeetonair + rew_knee
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # print(rew_gait_pattern[0])

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["collision"] += rew_collision
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["air_time"] += rew_airTime
        self.episode_sums["gait_pattern"] += rew_gait_pattern
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["spine"] += rew_spine
        self.episode_sums["hip"] += rew_hip
        self.episode_sums["power"] += rew_power
        self.episode_sums["allfeetonair"] += rew_allfeetonair
        self.episode_sums["thigh"] += rew_thigh
        self.episode_sums["knee"] += rew_knee

    def reset_idx(self, env_ids):
        positions_offset = torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.individual_step_counter[env_ids] = 0


        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        if self.command_x_range[1] > 1:
            self.commands[env_ids, 1] *= (torch.clip(self.command_x_range[1]-self.commands[env_ids, 0],0,self.command_x_range[1]-1)/(self.command_x_range[1]-1))
            self.commands[env_ids, 3] *= (torch.clip(self.command_x_range[1]-self.commands[env_ids, 0],0,self.command_x_range[1]-1)/(self.command_x_range[1]-1))
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :4], dim=1) > 0.25).unsqueeze(1) # set small commands to zero
        self.commands_init[:, 0] = self.commands[:, 0]
        self.commands_init[:, 1] = self.commands[:, 1]
        self.commands_init[:, 2] = self.commands[:, 2]
        # gallop_flag = (torch.norm(self.commands[env_ids, 0]) > 2).unsqueeze(0)
        # self.gait_pattern[env_ids]=torch.randint(0, 3, (len(env_ids), 1), out=self.gait_pattern[env_ids], device=self.device)
        
        # self.gait_pattern[env_ids,0] = torch.clip(self.gait_pattern[env_ids,0] + 4*gallop_flag.squeeze(), 0, 3 )
        # print(self.gait_pattern)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def push_robots(self):
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions[..., (0,3,8,11)] *= self.action_scale_hip
        self.actions[..., (1,4,9,12)] *= self.action_scale_thigh
        self.actions[..., (2,5,10,13)] *= self.action_scale_knee
        self.actions[..., (6,7)] *= self.action_scale_spine

        torques=torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        for i in range(self.decimation):
            # torques[...,(0,1,2,3,4,5,8,9,10,11,12,13)] = torch.clip(self.Kp_leg*(self.actions[...,(0,1,2,3,4,5,8,9,10,11,12,13)] + self.default_dof_pos[...,(0,1,2,3,4,5,8,9,10,11,12,13)] - self.dof_pos[...,(0,1,2,3,4,5,8,9,10,11,12,13)]) - self.Kd_leg*self.dof_vel[...,(0,1,2,3,4,5,8,9,10,11,12,13)],
            #                      -20., 20.)
            # torques[...,(6,7)] = torch.clip(self.Kp_spine*(self.actions[...,(6,7)] + self.default_dof_pos[...,(6,7)] - self.dof_pos[...,(6,7)]) - self.Kd_spine*self.dof_vel[...,(6,7)],
            #                      -20., 20.)
            torques = torch.clip(self.actions, -20, 20)
            # print(torques)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        self.individual_step_counter += 1
        
        if self.common_step_counter % self.push_interval == 0:
            self.push_robots()

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        heading_yaw=torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        turning_signal = torch.clip(self.individual_step_counter-self.turning_time, 0, 1)
        heading_yaw = turning_signal * (self.commands[:, 3])
        # heading_yaw = self.commands[:, 3]


        # self.commands[:, 2] = heading_yaw
        # self.commands[:, 2] = torch.clip(wrap_to_pi(heading_yaw - heading), -3.14, 3.14)

        self.commands_init[:, 2] = torch.clip(2*wrap_to_pi(heading_yaw - heading), -3.14, 3.14)
        self.commands_buf[:, 0, self.command_counter] = self.base_lin_vel[:, 0]
        self.commands_buf[:, 1, self.command_counter] = self.base_lin_vel[:, 1]
        self.commands_buf[:, 2, self.command_counter] = self.base_ang_vel[:, 2]
        self.command_counter += 1
        if self.common_step_counter % self.command_adjustment_rate == 0:
            commands_error_x = self.commands_init[...,0] - torch.mean(self.commands_buf[..., 0, :], dim=1)
            commands_error_y = self.commands_init[...,1] - torch.mean(self.commands_buf[..., 1, :], dim=1)
            commands_error_yaw = self.commands_init[...,2] - torch.mean(self.commands_buf[..., 2, :], dim=1)
            self.commands_buf = torch.zeros(self.num_envs, 3, self.command_adjustment_rate, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel
            self.commands[..., 0] = self.commands_init[..., 0] + commands_error_x * 0.1
            self.commands[..., 1] = self.commands_init[..., 1] + commands_error_y * 0.1
            self.commands[..., 2] = self.commands_init[..., 2] + commands_error_yaw * 0.1
            self.command_counter = 0
            # print(self.commands)
        


        
        # print(self.commands[0, 2])
        # print(self.base_ang_vel[0,0])

        ########saving data####################
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        data=np.zeros(27)
        data[0]=self.base_lin_vel[0,0]
        data[1]=self.base_lin_vel[0,1]
        data[2]=self.base_lin_vel[0,2]
        data[3]=self.base_ang_vel[0,0]
        data[4]=self.base_ang_vel[0,1]
        data[5]=self.base_ang_vel[0,2]
        data[6]=self.commands[0, 0]
        data[7]=self.commands[0, 1]
        data[8]=self.commands[0, 2]
        data[9]=self.torques[0, 0]
        data[10]=self.torques[0, 1]
        data[11]=self.torques[0, 2]
        data[12]=self.torques[0, 3]
        data[13]=self.torques[0, 4]
        data[14]=self.torques[0, 5]
        data[15]=self.torques[0, 6]
        data[16]=self.torques[0, 7]
        data[17]=self.torques[0, 8]
        data[18]=self.torques[0, 9]
        data[19]=self.torques[0, 10]
        data[20]=self.torques[0, 11]
        data[21]=self.torques[0, 12]
        data[22]=self.torques[0, 13]
        data[23]=contact[0,0]
        data[24]=contact[0,1]
        data[25]=contact[0,2]
        data[26]=contact[0,3]
        for i in range(27):
            data[i]=round(data[i],3)
        header=['lin_vel_x','lin_vel_y','lin_vel_z','ang_vel_x','ang_vel_y','ang_vel_z', 'command_x', 'command_y', 'command_yaw',
                'torque_hip_LF','torque_thigh_LF','torque_knee_LF','torque_hip_RF','torque_thigh_RF','torque_knee_RF',
                'torque_spine_roll','torque_spine_pitch',
                'torque_hip_LH','torque_thigh_LH','torque_knee_LH','torque_hip_RH','torque_thigh_RH','torque_knee_RH',
                'contact_LF','contact_RF','contact_LH','contact_RH']
        with open('data/temp.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if os.stat('data/temp.csv').st_size == 0:
                writer.writerow(header)
            writer.writerow(data)
        #######################################
            
        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0]+self.root_states[0, 0], p[1]+self.root_states[0, 1], p[2]+self.root_states[0, 2])
            cam_target = gymapi.Vec3(p[0]+self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]-0.1)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

####################################################################################
        # self.render_fps = 2
        # time.sleep(np.abs(1/self.render_fps))
        # print(self.phase_time)
####################################################################################

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False) # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


# terrain generator
from isaacgym.terrain_utils import *
class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
    
    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                difficulty = 1 / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.5
                step_height = 0.025 + 0.1 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.1
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.5, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                #     stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles
