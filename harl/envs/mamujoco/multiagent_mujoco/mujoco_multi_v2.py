from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .manyagent_swimmer import ManyAgentSwimmerEnv
from .obsk import get_joints_at_kdist, get_parts_and_edges, build_obs


def env_fn(env, **kwargs) -> MultiAgentEnv:  # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


env_REGISTRY = {}
env_REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)


# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        action = (action + 1) / 2
        action *= self.action_space.high - self.action_space.low
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= self.action_space.high - self.action_space.low
        action = action * 2 - 1
        return action


class MujocoMulti(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"]  # e.g. Ant-v2
        self.agent_conf = kwargs["env_args"]["agent_conf"]  # e.g. '2x3'

        (
            self.agent_partitions,
            self.mujoco_edges,
            self.mujoco_globals,
        ) = get_parts_and_edges(self.scenario, self.agent_conf)

        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs["env_args"].get("obs_add_global_pos", False)

        self.agent_obsk = kwargs["env_args"].get(
            "agent_obsk", None
        )  # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs["env_args"].get(
            "agent_obsk_agents", False
        )  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            self.k_categories_label = kwargs["env_args"].get("k_categories")
            if self.k_categories_label is None:
                if self.scenario in ["Ant-v2", "manyagent_ant"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext|qpos"
                elif self.scenario in ["Humanoid-v2", "HumanoidStandup-v2"]:
                    self.k_categories_label = (
                        "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
                    )
                elif self.scenario in ["Reacher-v2"]:
                    self.k_categories_label = "qpos,qvel,fingertip_dist|qpos"
                elif self.scenario in ["coupled_half_cheetah"]:
                    self.k_categories_label = "qpos,qvel,ten_J,ten_length,ten_velocity|"
                else:
                    self.k_categories_label = "qpos,qvel|qpos"

            k_split = self.k_categories_label.split("|")
            self.k_categories = [
                k_split[k if k < len(k_split) else -1].split(",")
                for k in range(self.agent_obsk + 1)
            ]

            self.global_categories_label = kwargs["env_args"].get("global_categories")
            self.global_categories = (
                self.global_categories_label.split(",")
                if self.global_categories_label is not None
                else []
            )

        if self.agent_obsk is not None:
            self.k_dicts = [
                get_joints_at_kdist(
                    agent_id,
                    self.agent_partitions,
                    self.mujoco_edges,
                    k=self.agent_obsk,
                    kagents=False,
                )
                for agent_id in range(self.n_agents)
            ]

        # load scenario from script
        self.episode_limit = self.args.episode_limit

        self.env_version = kwargs["env_args"].get("env_version", 2)
        if self.env_version == 2:
            try:
                self.wrapped_env = NormalizedActions(gym.make(self.scenario))
            except gym.error.Error:
                self.wrapped_env = NormalizedActions(
                    TimeLimit(
                        partial(env_REGISTRY[self.scenario], **kwargs["env_args"])(),
                        max_episode_steps=self.episode_limit,
                    )
                )
        else:
            assert False, "not implemented!"
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.episode_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = self.get_obs_size()
        self.share_obs_size = self.get_state_size()

        # COMPATIBILITY
        self.n = self.n_agents
        # self.observation_space = [Box(low=np.array([-10]*self.n_agents), high=np.array([10]*self.n_agents)) for _ in range(self.n_agents)]
        self.observation_space = [
            Box(low=-10, high=10, shape=(self.obs_size,)) for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-10, high=10, shape=(self.share_obs_size,))
            for _ in range(self.n_agents)
        ]

        acdims = [len(ap) for ap in self.agent_partitions]
        self.action_space = tuple(
            [
                Box(
                    self.env.action_space.low[sum(acdims[:0]) : sum(acdims[: 1])],
                    self.env.action_space.high[sum(acdims[:0]) : sum(acdims[: 1])],
                )
                for a in range(self.n_agents)
            ]
        )
        self.true_action_space = tuple(
            [
                Box(
                    self.env.action_space.low[sum(acdims[:a]) : sum(acdims[: a + 1])],
                    self.env.action_space.high[sum(acdims[:a]) : sum(acdims[: a + 1])],
                )
                for a in range(self.n_agents)
            ]
        )
        # === 添加滑动检测功能 ===
        env_args = kwargs.get("env_args", {})
        self.enable_slip_detection = env_args.get('enable_slip_detection', False)
        
        if self.enable_slip_detection:
            self.slip_penalty_weight = env_args.get('slip_penalty_weight', 0.1)
            self.contact_force_threshold = env_args.get('contact_force_threshold', 1.0)
            self.max_slip_velocity = env_args.get('max_slip_velocity', 0.1)
            self.max_slip_ratio = env_args.get('max_slip_ratio', 5.0)
            
            # 初始化滑动检测状态
            self.last_foot_positions = None
            self.feet_indices = self._get_feet_indices()
            
            print(f"滑动惩罚已启用 - 权重: {self.slip_penalty_weight}")
        else:
            print("滑动惩罚未启用")
        pass

    def step(self, actions):
        # need to remove dummy actions that arise due to unequal action vector sizes across agents
        flat_actions = np.concatenate(
            [
                actions[i][: self.true_action_space[i].low.shape[0]]
                for i in range(self.n_agents)
            ]
        )
        obs_n, reward_n, done_n, info_n = self.wrapped_env.step(flat_actions)
        self.steps += 1

        info = {}
        info.update(info_n)
        # === 在这里添加滑动惩罚计算 ===
        slip_penalty = 0.0
        if hasattr(self, 'enable_slip_detection') and self.enable_slip_detection:
            slip_penalty = self._calculate_slip_penalty_humanplus_style()
            info['slip_penalty'] = slip_penalty
            info['slip_detected'] = slip_penalty > 0.001
        # if done_n:
        #     if self.steps < self.episode_limit:
        #         info["episode_limit"] = False   # the next state will be masked out
        #     else:
        #         info["episode_limit"] = True    # the next state will not be masked out
        if done_n:
            if self.steps < self.episode_limit:
                # the next state will be masked out
                info["bad_transition"] = False
            else:
                # the next state will not be masked out
                info["bad_transition"] = True

        # return reward_n, done_n, info
        # === 修改奖励计算 ===
        modified_reward = reward_n - slip_penalty  # 添加这一行
        rewards = [[modified_reward]] * self.n_agents # 修改这一行（原来是reward_n）
        dones = [done_n] * self.n_agents
        infos = [info for _ in range(self.n_agents)]
        return (
            self.get_obs(),
            self.get_state(),
            rewards,
            dones,
            infos,
            self.get_avail_actions(),
        )

    def get_obs(self):
        """Returns all agent observat3ions in a list"""
        state = self.env._get_obs()
        obs_n = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            # obs_n.append(self.get_obs_agent(a))
            # obs_n.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
            # obs_n.append(np.concatenate([self.get_obs_agent(a), agent_id_feats]))
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs_n.append(obs_i)
        return obs_n

    def get_obs_agent(self, agent_id):
        if self.agent_obsk is None:
            return self.env._get_obs()
        else:
            # return build_obs(self.env,
            #                       self.k_dicts[agent_id],
            #                       self.k_categories,
            #                       self.mujoco_globals,
            #                       self.global_categories,
            #                       vec_len=getattr(self, "obs_size", None))
            return build_obs(
                self.env,
                self.k_dicts[agent_id],
                self.k_categories,
                self.mujoco_globals,
                self.global_categories,
            )

    def get_obs_size(self):
        """Returns the shape of the observation"""
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return len(self.get_obs()[0])
            # return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])

    def get_state(self, team=None):
        # TODO: May want global states for different teams (so cannot see what the other team is communicating e.g.)
        state = self.env._get_obs()
        state_normed = (state - np.mean(state)) / np.std(state)
        share_obs = []
        for a in range(self.n_agents):
            share_obs.append(state_normed)
        return share_obs

    def get_state_size(self):
        """Returns the shape of the state"""
        return len(self.get_state()[0])

    def get_avail_actions(self):  # all actions are always available
        # return np.ones(shape=(self.n_agents, self.n_actions,))
        return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        # return np.ones(shape=(self.n_actions,))
        return None

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return (
            self.n_actions
        )  # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        """Returns initial observations and states"""
        self.steps = 0
        self.timelimit_env.reset()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    def seed(self, seed):
        self.wrapped_env.seed(seed)

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_space,
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }
        return env_info
    # === 添加滑动惩罚计算函数 ===
    # === 以下是新添加的滑动检测方法 ===
    
    def _get_feet_indices(self):
        """获取足部索引"""
        if not hasattr(self, 'env') or self.env is None:
            return [0, 1]  # 默认值
            
        sim = self.env.sim
        feet_names = ['left_foot', 'right_foot', 'foot_left', 'foot_right', 
                     'ankle_left', 'ankle_right', 'l_ankle', 'r_ankle']
        
        feet_indices = []
        for foot_name in feet_names:
            try:
                foot_id = sim.model.body_name2id(foot_name)
                feet_indices.append(foot_id)
                print(f"找到足部: {foot_name} -> ID: {foot_id}")
            except:
                continue
        
        if not feet_indices:
            # 如果找不到足部，尝试查找包含'foot'或'ankle'的所有body
            try:
                for i in range(sim.model.nbody):
                    body_name = sim.model.body_id2name(i)
                    if body_name and ('foot' in body_name.lower() or 'ankle' in body_name.lower()):
                        feet_indices.append(i)
                        print(f"自动发现足部: {body_name} -> ID: {i}")
            except:
                pass
        
        return feet_indices if feet_indices else [0, 1]  # 默认值

    def _calculate_slip_penalty_humanplus_style(self):
        """基于humanplus风格的滑动惩罚计算"""
        try:
            # 获取接触力
            contact_forces = self._get_contact_forces()
            if contact_forces is None or len(contact_forces) == 0:
                return 0.0
            
            # 获取足部位置用于速度计算
            current_foot_positions = self._get_foot_positions()
            
            # 基于接触力的滑动检测
            force_slip_penalty = self._detect_force_based_slip(contact_forces)
            
            # 基于速度的滑动检测
            velocity_slip_penalty = self._detect_velocity_based_slip(current_foot_positions)
            
            # 综合惩罚
            total_penalty = (force_slip_penalty + velocity_slip_penalty) * self.slip_penalty_weight
            
            return total_penalty
            
        except Exception as e:
            print(f"滑动检测计算错误: {e}")
            return 0.0

    def _get_contact_forces(self):
    	#"""获取足部接触力 - 兼容mujoco_py版本"""
        if not hasattr(self, 'env') or self.env is None:
            return {}
        
        sim = self.env.sim
        contact_forces = {}
    
        for foot_idx in self.feet_indices:
            contact_forces[foot_idx] = np.zeros(3)
    
        try:
            # 遍历所有接触点
            for i in range(sim.data.ncon):
                contact = sim.data.contact[i]
            
                # 检查接触是否涉及足部
                geom1_body = sim.model.geom_bodyid[contact.geom1]
                geom2_body = sim.model.geom_bodyid[contact.geom2]
            
                for foot_idx in self.feet_indices:
                    if geom1_body == foot_idx or geom2_body == foot_idx:
                        # 使用正确的mujoco_py API获取接触力
                        # 计算接触力需要从接触信息中推导
                        c_array = np.zeros(6, dtype=np.float64)
                    
                        # 使用mujoco_py的函数计算接触力
                        from mujoco_py import functions
                        functions.mj_contactForce(sim.model, sim.data, i, c_array)
                    
                        # c_array前3个元素是法向和切向力
                        force = c_array[:3]
                        contact_forces[foot_idx] += force
                    
        except Exception as e:
            print(f"获取接触力错误: {e}")
            # 如果接触力获取失败，使用替代方案
            return self._get_contact_forces_fallback()
    
        return contact_forces

    def _get_contact_forces_fallback(self):
        """接触力获取失败时的备用方案"""
        contact_forces = {}
    
        try:
            sim = self.env.sim
        
            # 使用传感器数据作为替代（如果可用）
            if hasattr(sim.data, 'sensordata') and sim.data.sensordata.size > 0:
                # 尝试从传感器数据中获取力信息
                sensor_data = sim.data.sensordata
            
                for i, foot_idx in enumerate(self.feet_indices):
                    if i * 3 + 2 < len(sensor_data):
                        force = sensor_data[i*3:(i+1)*3]
                        contact_forces[foot_idx] = force
                    else:
                        contact_forces[foot_idx] = np.zeros(3)
            else:
                # 如果没有传感器数据，使用关节力矩作为近似
                for foot_idx in self.feet_indices:
                    contact_forces[foot_idx] = np.zeros(3)
                
        except Exception as e:
            print(f"备用接触力获取方法也失败: {e}")
            # 最后的兜底方案
            for foot_idx in self.feet_indices:
                contact_forces[foot_idx] = np.zeros(3)
    
        return contact_forces

    def _detect_force_based_slip(self, contact_forces):
        """基于接触力的滑动检测（参考humanplus stumble函数）"""
        total_penalty = 0.0
        
        for foot_idx, force in contact_forces.items():
            # 计算水平和垂直力
            horizontal_force = np.linalg.norm(force[:2])
            vertical_force = abs(force[2])
            
            # 检查是否在接触状态
            if vertical_force > self.contact_force_threshold:
                # 计算滑动比例（参考humanplus中的5倍阈值）
                slip_ratio = horizontal_force / vertical_force if vertical_force > 0 else 0
                
                if slip_ratio > self.max_slip_ratio:
                    # 计算惩罚强度
                    slip_intensity = slip_ratio - self.max_slip_ratio
                    total_penalty += slip_intensity ** 2
        
        return total_penalty

    def _detect_velocity_based_slip(self, current_positions):
        """基于速度的滑动检测"""
        if self.last_foot_positions is None:
            self.last_foot_positions = current_positions
            return 0.0
        
        dt = getattr(self.env, 'dt', 0.002)
        total_penalty = 0.0
        
        try:
            for i, foot_idx in enumerate(self.feet_indices):
                if i < len(current_positions) and i < len(self.last_foot_positions):
                    # 计算水平速度
                    velocity = (current_positions[i] - self.last_foot_positions[i]) / dt
                    horizontal_velocity = np.linalg.norm(velocity[:2])
                    
                    # 检查是否超过阈值
                    if horizontal_velocity > self.max_slip_velocity:
                        excess_velocity = horizontal_velocity - self.max_slip_velocity
                        total_penalty += excess_velocity ** 2
        except Exception as e:
            print(f"速度滑动检测错误: {e}")
        
        # 更新位置记录
        self.last_foot_positions = current_positions
        
        return total_penalty

    def _get_foot_positions(self):
        """获取足部位置"""
        if not hasattr(self, 'env') or self.env is None:
            return np.zeros((len(self.feet_indices), 3))
            
        sim = self.env.sim
        positions = []
        
        for foot_idx in self.feet_indices:
            try:
                pos = sim.data.body_xpos[foot_idx].copy()
                positions.append(pos)
            except:
                positions.append(np.zeros(3))
        
        return np.array(positions)
