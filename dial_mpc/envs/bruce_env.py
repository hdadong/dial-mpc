from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model

import mujoco
from mujoco import mjx

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step
from dial_mpc.utils.io_utils import get_model_path


@dataclass
class BruceWalkEnvConfig(BaseEnvConfig):
    kp: Union[float, jax.Array] = jnp.array(
        [
            7.0,
            10.0,
            7.0,  # left hips
            10.0,
            1.5,  # left knee, ankle
            7.0,
            10.0,
            7.0,  # left hips
            10.0,
            1.5,  # left knee, ankle
        ]
    )
    kd: Union[float, jax.Array] = jnp.array(
        [
            0.2,
            0.4,
            0.2,  # left hips
            0.4,
            0.08,  # left knee, ankle
            0.2,
            0.4,
            0.2,  # left hips
            0.4,
            0.08,  # left knee, ankle
        ]
    )
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "jog"


class BruceWalkEnv(BaseEnv):
    def __init__(self, config: BruceWalkEnvConfig):
        super().__init__(config)
        print("haha",mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "ankle_pitch_r"
        ))
        print(
            mujoco.mj_name2id(
                        self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "hip_yaw_r"
                    )
        )
        print(mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "bruce-pelvis"
        ))
        print(mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "toe_right_collision"
        ))
        # some body indices
        self._pelvis_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "bruce-pelvis"
        )
        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "bruce-pelvis"
        )

        self._left_foot_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "lf_sole"
        )
        self._right_foot_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "rf_sole"
        )
        self._feet_site_id = jnp.array(
            [self._left_foot_idx, self._right_foot_idx], dtype=jnp.int32
        )
        # gait phase
        self._gait = config.gait
        self._gait_phase = {
            "stand": jnp.zeros(2),
            "slow_walk": jnp.array([0.0, 0.5]),
            "walk": jnp.array([0.0, 0.5]),
            "jog": jnp.array([0.0, 0.5]),
        }
        self._gait_params = {
            # ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "slow_walk": jnp.array([0.6, 0.8, 0.15]),
            "walk": jnp.array([0.5, 1.0, 0.15]),
            "jog": jnp.array([0.15, 1, 0.1]),
        }

        # joint limits and initial pose
        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]
        # joint sampling range
        self.joint_range = jnp.array(
            [
                [-0.8, 0.8],
                [-0.4, 2.5],
                [-0.8, 0.8],
                [-2.5, 0.25],
                [-0.8, 0.8],

                [-0.8, 0.8],
                [-0.4, 2.5],
                [-0.8, 0.8],
                [-2.5, 0.25],
                [-0.8, 0.8],

            ]
        )
        # self.joint_range = self.physical_joint_range
        self._period = 0.55
        self.roll_yaw = jnp.array([0, 2, 5, 7])
        print("self.dt", self.dt)
    def make_system(self, config: BruceWalkEnvConfig) -> System:
        model_path = get_model_path("bruce", "scene.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def compute_ref_state(self, sin_pos: jnp.ndarray):
        sin_pos_l = sin_pos.copy()
        sin_pos_r = sin_pos.copy()

        pdes = jnp.zeros((self.action_size,))
        
        # left stand
        sin_pos_l = jnp.where(sin_pos_l > 0, 0.0, sin_pos_l)
        sin_pos_r = jnp.where(sin_pos_r < 0, 0.0, sin_pos_r)

        scale_1 = 0.3
        scale_2 = 0.6
        scale_3 = 0.3
        # left stand
        pdes = pdes.at[1].set(-(sin_pos_l) * scale_1)
        pdes = pdes.at[3].set(sin_pos_l * scale_2)
        pdes = pdes.at[4].set(-sin_pos_l * scale_3)
        
        # right swing
        pdes = pdes.at[6].set(sin_pos_r * scale_1)
        pdes = pdes.at[8].set(-sin_pos_r * scale_2)
        pdes = pdes.at[9].set(sin_pos_r * scale_3)

        pdes = jnp.where(jnp.abs(sin_pos) < 0.1, 0.0, pdes)

        pdes += self._default_pose
        contact = self._get_gait_phase(sin_pos)

        return pdes, contact

    def _get_gait_phase(self, sin_curve):
        # return float mask 1 is stance, 0 is swing
        stance_mask = jnp.array([sin_curve >= 0, sin_curve < 0])

        # Double support phase
        stance_mask = jnp.where(jnp.abs(sin_curve) < 0.1, True, stance_mask)

        return stance_mask

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.0, 0.0, 0.49]),
            "vel_tar": jnp.array([0.2, 0.0, 0.0]),
            "ang_vel_tar": jnp.zeros(3),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(2),
            "z_feet_tar": jnp.zeros(2),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(2, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(2),
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # switch to new target if randomize_target is True
        def dont_randomize():
            return (
                jnp.array([self._config.default_vx, self._config.default_vy, 0.0]),
                jnp.array([0.0, 0.0, self._config.default_vyaw]),
            )

        def randomize():
            return self.sample_command(cmd_rng)

        _, _ = jax.lax.cond(
            (state.info["randomize_target"]) & (state.info["step"] % 500 == 0),
            randomize,
            dont_randomize,
        )
        vel_tar = state.info["vel_tar"]
        ang_vel_tar = state.info["ang_vel_tar"]

        # state.info["vel_tar"] = jnp.minimum(
        #     vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time, vel_tar
        # )
        # state.info["ang_vel_tar"] = jnp.minimum(
        #     ang_vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time,
        #     ang_vel_tar,
        # )

        # reward
        # gaits reward
        # z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        # reward_gaits = -jnp.sum(((z_feet_tar - z_feet)) ** 2)
        z_feet = jnp.array(
            [
                jnp.min(pipeline_state.contact.dist[3:5]),
                jnp.min(pipeline_state.contact.dist[7:9]),
            ]
        )
        reward_gaits = -jnp.sum((z_feet_tar - z_feet) ** 2)
        # foot contact data based on z-position
        # pytype: disable=attribute-error
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2]
        contact = foot_contact_z < 0.026  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 0.026) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt
        reward_air_time = jnp.sum((state.info["feet_air_time"] - 0.1) * first_contact)
        # position reward
        pos_tar = (
            state.info["pos_tar"] + state.info["vel_tar"] * self.dt * state.info["step"]
        )
        pos = x.pos[self._torso_idx - 1]
        reward_pos = -jnp.sum((pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = (
            state.info["yaw_tar"]
            + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        )
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        # stay to norminal pose reward
        reward_pose = -jnp.sum(jnp.square(joint_targets[self.roll_yaw] - self._default_pose[self.roll_yaw]))
        # velocity reward
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)
        # height reward
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        )
        # energy reward
        # reward_energy = -jnp.sum((ctrl * pipeline_state.qvel[6:] / 160.0) ** 2)
        reward_energy = -jnp.sum((ctrl / self.joint_torque_range[:, 1]) ** 2)
        # stay alive reward
        reward_alive = 1.0 - state.done

        t = state.info["step"] * self.dt
        phase = 2*jnp.pi*t/self._period
        ref_action, stance_mask = self.compute_ref_state(jnp.sin(phase))
        joint_pos = joint_targets
        pos_target = ref_action
        diff = joint_pos - pos_target
        norm_diff = jnp.linalg.norm(diff)
        reward_ref = jnp.exp(-2 * norm_diff) - 0.2 * jnp.clip(norm_diff, 0, 0.5)
        

        # feet_reward
        fd, max_df = 0.15, 0.2
        foot_pos = x.pos[jnp.array([5, 10]), :2]
        foot_dist = jnp.linalg.norm(foot_pos[:] - foot_pos[:])
        d_min = jnp.clip(foot_dist - fd, -0.152, 0.0)
        d_max = jnp.clip(foot_dist - max_df, 0.0, 0.152)
        reward_feet_distance = (jnp.exp(-jnp.abs(d_min) * 100) + jnp.exp(-jnp.abs(d_max) * 100)) / 2

        # knee reward
        knee_pos = x.pos[jnp.array([4, 9]), :2]
        knee_dist = jnp.linalg.norm(knee_pos[0, :] - knee_pos[1, :])
        d_min = jnp.clip(knee_dist - fd, -0.152, 0.0)
        d_max = jnp.clip(knee_dist - max_df, 0.0, 0.152)
        reward_knee_distance = (jnp.exp(-jnp.abs(d_min) * 100) + jnp.exp(-jnp.abs(d_max) * 100)) / 2
        
        # reward
        reward = (
            #reward_gaits * 5.0
            #+ reward_feet_distance * 0.5
            #+ reward_knee_distance * 0.1
            + reward_ref * 1.0
            #+ reward_air_time * 1.0
            #+ reward_pos * 0.0
            + reward_upright * 0.5
           # + reward_yaw * 0.1
            # + reward_pose * 0.5
           # + reward_vel * 0.5
           # + reward_ang_vel * 1.0
            + reward_height * 0.5
            + reward_energy * 0.01
            + reward_alive * 0.0
        )

        # done
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        joint_angles = joint_angles[: len(self.joint_range)]
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        obs = jnp.concatenate(
            [
                state_info["vel_tar"],
                state_info["ang_vel_tar"],
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb,
                ab,
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)

    def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        lin_vel_x = [-1.5, 1.5]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_lin_vel_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], 0.0])
        new_ang_vel_cmd = jnp.array([0.0, 0.0, ang_vel_yaw[0]])
        return new_lin_vel_cmd, new_ang_vel_cmd


brax_envs.register_environment("bruce_walk", BruceWalkEnv)
