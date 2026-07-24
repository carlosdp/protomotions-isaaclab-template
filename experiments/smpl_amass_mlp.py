# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Downstream SMPL AMASS MLP mimic experiment for installed ProtoMotions.

This project-local experiment mirrors the behavior of ProtoMotions'
``examples/experiments/mimic/mlp.py`` without importing the unshipped
``examples`` tree. The ProtoMotions CLI loads this file by path while all
framework, simulator, robot, environment, and agent code comes from the
installed distribution.
"""

import argparse

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


def terrain_config(args: argparse.Namespace) -> TerrainConfig:
    """Use the upstream flat-terrain defaults."""
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace) -> SceneLibConfig:
    """Load an optional scene file supplied on the ProtoMotions CLI."""
    return SceneLibConfig(scene_file=getattr(args, "scenes_file", None))


def motion_lib_config(args: argparse.Namespace) -> MotionLibConfig:
    """Load the external, packaged AMASS MotionLib file."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build the full-body mimic environment used by the upstream MLP run."""
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        contact_match_rew_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        previous_actions_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        control_components={
            "mimic": MimicControlConfig(bootstrap_on_episode_end=True),
        },
        observation_components={
            "max_coords_obs": max_coords_obs_factory(),
            "previous_actions": previous_actions_factory(history_steps=1),
            "mimic_target_poses": mimic_target_poses_max_coords_factory(
                with_velocities=True
            ),
        },
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.5),
        },
        reward_components={
            "action_smoothness": action_smoothness_factory(weight=-0.02),
            **mimic_tracking_rewards_factory(
                gt_weight=0.5,
                gr_weight=0.3,
                gv_weight=0.1,
                gav_weight=0.2,
                rh_weight=0.2,
                gt_coef=-25.0,
                gr_coef=-5.0,
                gv_coef=-0.5,
                gav_coef=-0.1,
                rh_coef=-100.0,
            ),
            "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
            "contact_match_rew": contact_match_rew_factory(
                weight=-0.1,
                zero_during_grace_period=True,
            ),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> PPOAgentConfig:
    """Build the upstream six-layer actor and four-layer critic."""
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        AdvantageNormalizationConfig,
        PPOActorConfig,
        PPOModelConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    input_keys = ["max_coords_obs", "mimic_target_poses", "previous_actions"]
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=input_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=input_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )
    critic_config = MLPWithConcatConfig(
        in_keys=input_keys,
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=input_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True,
            shift_mean=True,
            use_ema=True,
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
) -> None:
    """Enable the SMPL foot contact sensors used by the contact reward."""
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: PPOAgentConfig,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
) -> None:
    """Match the upstream full-motion evaluation behavior."""
    if env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1_000_000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
