import numpy as np

from gym import utils
from envs.mujoco_env import MujocoEnv

from gym.spaces import Box
import ipdb
from scipy.interpolate import interp1d
from envs.humanoid.humanoid_utils import flip_action, flip_position, flip_velocity
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidTreadmillEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": -1,  # redefined
    }

    def __init__(self, healthy_z_range=(1.0, 4.0), args=None, **kwargs):
        utils.EzPickle.__init__(self, healthy_z_range, args, **kwargs)
        self.args = args

        self._healthy_z_range = healthy_z_range

        # Gait
        ref = np.loadtxt(args.gait_ref_path)

        self.gait_ref = interp1d(np.arange(0, ref.shape[0]) / (ref.shape[0] - 1), ref, axis=0)
        self.treadmill_velocity = args.treadmill_velocity
        self.gait_cycle_time = args.gait_cycle_time

        # Incrementors and counters
        self.initial_phase_offset = np.random.randint(0, 50) / 50
        self.action_phase_offset = 0
        self.pert_cntr = 0

        observation_space = Box(low=-np.inf, high=np.inf, shape=(args.agent_obs_size,), dtype=np.float64)

        MujocoEnv.__init__(self, args.xml_file, args.frame_skip, observation_space=observation_space, **kwargs)

        self.time_step = self.model.opt.timestep

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy
        return terminated

    @property
    def phase(self):
        initial_offset_time = self.initial_phase_offset * self.gait_cycle_time
        action_offset_time = self.action_phase_offset * self.gait_cycle_time
        total_time = initial_offset_time + action_offset_time + self.data.time

        return (total_time % self.gait_cycle_time) / self.gait_cycle_time

    @property
    def target_reference(self):
        frame_skip_time = self.frame_skip * self.time_step
        initial_offset_time = self.initial_phase_offset * self.gait_cycle_time
        action_offset_time = self.action_phase_offset * self.gait_cycle_time
        total_time = frame_skip_time + initial_offset_time + action_offset_time + self.data.time

        phase_target = (total_time % self.gait_cycle_time) / self.gait_cycle_time
        gait_ref = self.gait_ref(phase_target)

        return gait_ref

    def _get_obs(self):
        if self.phase <= .5:
            position = self.data.qpos.flat.copy()
            position = position[:-1]  # exclude treadmill

            velocity = np.clip(self.data.qvel.flat.copy()[:-1], -1000, 1000)
            velocity[0] *= 10
            velocity[1] *= 10
            velocity[2] *= 10

            observation = np.concatenate((position, velocity / 10, np.array([self.phase]))).ravel()
        else:
            position = self.data.qpos.flat.copy()
            position = position[:-1]  # exclude treadmill
            flipped_pos = flip_position(position)

            velocity = np.clip(self.data.qvel.flat.copy()[:-1], -1000, 1000)
            flipped_vel = flip_velocity(velocity)
            flipped_vel[0] *= 10
            flipped_vel[1] *= 10
            flipped_vel[2] *= 10

            observation = np.concatenate((flipped_pos, flipped_vel / 10, np.array([self.phase - .5]))).ravel()
        return observation

    def get_obs(self):
        return self._get_obs()

    def calc_reward(self, ref):
        joint_ref = ref[7:-1]
        joint_obs = self.data.qpos[7:-1].copy()
        joint_reward = np.exp(-2 * np.sum((joint_ref - joint_obs) ** 2))

        pos_ref = ref[0:3]
        pos = self.data.qpos[0:3]
        pos_reward = 1 * (0 - self.data.qvel[0]) ** 2 + 0.01 * (pos_ref[1] - pos[1]) ** 2 + (pos_ref[2] - pos[2]) ** 2
        pos_reward = np.exp(-1 * pos_reward)

        orient_ref = ref[3:7]
        orient_obs = self.data.qpos[3:7].copy()
        orient_reward = np.sum((orient_ref - orient_obs) ** 2) + 0.1 * np.sum((self.data.qvel[3:6]) ** 2)
        orient_reward = np.exp(-1 * orient_reward)

        reward = orient_reward * joint_reward * pos_reward * .1

        return reward

    def step(self, action):
        # ipdb.set_trace()
        action = np.array(action.tolist()).copy()

        joint_action = action[:-1] if self.phase <= .5 else flip_action(action[:-1])

        phase_action = self.args.phase_action_mag * action[-1]

        self.action_phase_offset += phase_action

        target_ref = self.target_reference.copy()
        final_target = joint_action + target_ref[7:-1]  # add action to joint ref to create final joint target

        grfc_info = None
        for _ in range(self.frame_skip):
            joint_obs = self.data.qpos[7:-1].copy()
            joint_vel_obs = self.data.qvel[6:-1].copy()

            error = final_target - joint_obs
            error_der = joint_vel_obs

            torque = 100 * error - 10 * error_der

            # GRF Data
            curr_torque = torque.copy()
            curr_state = self.get_obs()[:-1]

            left_force = self.data.body('left_ankle').cfrc_ext  # torque/force
            right_force = self.data.body('right_ankle').cfrc_ext

            grfc_row = np.hstack((curr_state, curr_torque, left_force, right_force))
            grfc_info = np.vstack((grfc_info, grfc_row)) if grfc_info is not None else grfc_row

            # GRF Application [force/torque ....]
            # self.data.body('left_ankle').xfrc_applied[2] = 500 # force/torque


            # Simulation
            self.data.qvel[-1] = -self.treadmill_velocity
            self.do_simulation(torque / 100, 1)
            # self.set_state(target_ref, self.init_qvel)

        self.renderer.render_step()

        observation = self._get_obs()
        reward = self.calc_reward(target_ref)

        terminated = self.terminated
        info = {}
        info['grfc_info'] = grfc_info

        done = False
        if self.data.time >= self.args.max_ep_time:
            done = True

        return observation, reward, terminated, done, info

    def reset_model(self):
        self.action_phase_offset = 0
        self.initial_phase_offset = np.random.randint(0, 50) / 50

        qpos = self.gait_ref(self.phase)
        self.set_state(qpos, self.init_qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
