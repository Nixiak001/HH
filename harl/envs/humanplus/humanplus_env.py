"""
HumanPlus HST Environment Wrapper for HARL.

This module provides the integration between HH (HARL) upper-level policy 
and HumanPlus HST (Humanoid Shadowing Transformer) lower-level controller.

Architecture (Residual Policy Learning):
    Upper Layer (HH/HARL): Outputs 19-dim action CORRECTIONS
    Lower Layer (HST): Generates baseline actions from pretrained policy
    Combined: final_action = HST_baseline + HH_correction
    Simulation: IsaacGym physics simulation

The upper-level HH policy learns to output corrections that improve upon
the pretrained HST controller's baseline behavior. This residual learning
approach allows HH to leverage HST's stable baseline while learning task-
specific improvements.
"""

import torch
import numpy as np
from gym import spaces


def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()


class HumanPlusEnv:
    """
    HumanPlus Environment wrapper for HARL framework.
    
    This environment wraps the HST (Humanoid Shadowing Transformer) environment
    from humanplus and exposes it to the HARL multi-agent RL framework.
    
    The key design is RESIDUAL POLICY LEARNING:
    - HST policy generates stable baseline actions
    - HH policy outputs corrections/modifications to these actions
    - Final action = HST_baseline + HH_correction
    
    This allows HH to learn improvements while maintaining HST's stability.
    
    Attributes:
        n_envs: Number of parallel environments
        n_agents: Number of agents (1 for single humanoid)
        num_dofs: Number of degrees of freedom (19 for H1 robot)
        observation_space: Observation space for each agent
        action_space: Action space for each agent (19-dim action corrections)
        share_observation_space: Shared observation space
    """
    
    def __init__(self, env_args):
        """
        Initialize the HumanPlus environment.
        
        Args:
            env_args: Dictionary containing environment configuration:
                - n_threads: Number of parallel environments
                - humanplus_path: Path to humanplus HST installation
                - headless: Whether to run without rendering
                - device: Device to run simulation on (cuda:0, cpu, etc.)
                - use_pretrained_hst: Whether to use pretrained HST policy
                - hst_checkpoint: Path to pretrained HST checkpoint
                - episode_length: Maximum episode length
                - num_dofs: Number of degrees of freedom (default: 19 for H1)
        """
        self.env_args = env_args
        self.n_envs = env_args.get("n_threads", 1)
        self.n_agents = 1  # Single humanoid agent
        # H1 robot has 19 DOFs, but make it configurable
        self.num_dofs = env_args.get("num_dofs", 19)
        
        # Device configuration
        self.device = env_args.get("device", "cuda:0")
        self.headless = env_args.get("headless", True)
        
        # HST configuration
        self.use_pretrained_hst = env_args.get("use_pretrained_hst", True)
        self.hst_checkpoint = env_args.get("hst_checkpoint", None)
        
        # Episode configuration
        self.episode_length = env_args.get("episode_length", 1000)
        self.max_episode_length = self.episode_length
        
        # Initialize the underlying HST environment
        self.env = self._create_hst_env(env_args)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Load pretrained HST if specified
        if self.use_pretrained_hst and self.hst_checkpoint:
            self._load_pretrained_hst()
        
        # Step counter
        self.current_step = 0
    
    def _create_hst_env(self, env_args):
        """
        Create the underlying HST environment.
        
        This method imports and initializes the H1 environment from humanplus.
        
        Args:
            env_args: Environment arguments
            
        Returns:
            Initialized HST environment
        """
        try:
            # Try to import humanplus HST environment
            import sys
            humanplus_path = env_args.get("humanplus_path", None)
            if humanplus_path:
                sys.path.insert(0, f"{humanplus_path}/HST/legged_gym")
                sys.path.insert(0, f"{humanplus_path}/HST/rsl_rl")
            
            from legged_gym.envs.h1.h1 import H1
            from legged_gym.envs.h1.h1_config import H1RoughCfg
            from isaacgym import gymapi
            
            # Create simulation parameters
            sim_params = gymapi.SimParams()
            sim_params.dt = H1RoughCfg.sim.dt
            sim_params.substeps = H1RoughCfg.sim.substeps
            sim_params.gravity = gymapi.Vec3(*H1RoughCfg.sim.gravity)
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.use_gpu_pipeline = True
            
            # PhysX parameters
            sim_params.physx.num_threads = H1RoughCfg.sim.physx.num_threads
            sim_params.physx.solver_type = H1RoughCfg.sim.physx.solver_type
            sim_params.physx.num_position_iterations = H1RoughCfg.sim.physx.num_position_iterations
            sim_params.physx.num_velocity_iterations = H1RoughCfg.sim.physx.num_velocity_iterations
            sim_params.physx.contact_offset = H1RoughCfg.sim.physx.contact_offset
            sim_params.physx.rest_offset = H1RoughCfg.sim.physx.rest_offset
            sim_params.physx.bounce_threshold_velocity = H1RoughCfg.sim.physx.bounce_threshold_velocity
            sim_params.physx.max_depenetration_velocity = H1RoughCfg.sim.physx.max_depenetration_velocity
            sim_params.physx.max_gpu_contact_pairs = H1RoughCfg.sim.physx.max_gpu_contact_pairs
            sim_params.physx.default_buffer_size_multiplier = H1RoughCfg.sim.physx.default_buffer_size_multiplier
            sim_params.physx.contact_collection = gymapi.ContactCollection(H1RoughCfg.sim.physx.contact_collection)
            
            # Update config with custom settings
            H1RoughCfg.env.num_envs = self.n_envs
            
            # CRITICAL FIX: Disable target_jt reward during hierarchical training!
            # The target_jt reward penalizes deviation from the npy trajectory,
            # which conflicts with HH learning its own control strategy.
            # Only keep task-relevant rewards (tracking_lin_vel, tracking_ang_vel).
            disable_target_jt = env_args.get("disable_target_jt_reward", True)
            if disable_target_jt:
                H1RoughCfg.rewards.scales.target_jt = 0.0
                print("Disabled target_jt reward for hierarchical training")
            
            # Store config for later use
            self.hst_cfg = H1RoughCfg
            self.obs_context_len = H1RoughCfg.env.obs_context_len  # Usually 8
            self.num_obs = H1RoughCfg.env.num_observations  # Usually 84
            
            # Create H1 environment
            env = H1(
                cfg=H1RoughCfg,
                sim_params=sim_params,
                physics_engine=gymapi.SIM_PHYSX,
                sim_device=self.device,
                headless=self.headless
            )
            
            # Update num_dofs from actual environment
            self.num_dofs = env.num_dofs if hasattr(env, 'num_dofs') else 19
            
            return env
            
        except ImportError as e:
            print(f"Warning: Could not import humanplus HST environment: {e}")
            print("Creating mock environment for development/testing...")
            self.obs_context_len = 8
            self.num_obs = 84
            return self._create_mock_env()
    
    def _create_mock_env(self):
        """
        Create a mock environment for testing without humanplus installation.
        
        Returns:
            Mock environment object
        """
        class MockHSTEnv:
            def __init__(self, num_envs, num_dofs, device, obs_context_len=8, num_obs=84):
                self.num_envs = num_envs
                self.num_dofs = num_dofs
                self.device = device
                self.num_obs = num_obs
                self.obs_context_len = obs_context_len
                
                # Mock state tensors
                self.obs_buf = torch.zeros(num_envs, self.num_obs, device=device)
                self.obs_history_buf = torch.zeros(num_envs, obs_context_len, num_obs, device=device)
                self.rew_buf = torch.zeros(num_envs, device=device)
                self.reset_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
                self.dof_pos = torch.zeros(num_envs, num_dofs, device=device)
                self.default_dof_pos = torch.zeros(1, num_dofs, device=device)
                self.target_jt = torch.zeros(num_envs, num_dofs, device=device)
                self.delayed_obs_target_jt = torch.zeros(num_envs, num_dofs, device=device)
                
            def reset(self):
                self.obs_buf = torch.randn(self.num_envs, self.num_obs, device=self.device) * 0.1
                self.obs_history_buf = self.obs_buf.unsqueeze(1).repeat(1, self.obs_context_len, 1)
                return self.obs_history_buf, None
            
            def step(self, actions):
                # Simulate one step
                self.obs_buf = torch.randn(self.num_envs, self.num_obs, device=self.device) * 0.1
                self.obs_history_buf = torch.cat([
                    self.obs_history_buf[:, 1:],
                    self.obs_buf.unsqueeze(1)
                ], dim=1)
                self.rew_buf = torch.ones(self.num_envs, device=self.device) * 0.1
                self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                extras = {}
                return self.obs_history_buf, None, self.rew_buf, self.reset_buf, extras
                
            def render(self):
                pass
        
        obs_context_len = getattr(self, 'obs_context_len', 8)
        num_obs = getattr(self, 'num_obs', 84)
        return MockHSTEnv(self.n_envs, self.num_dofs, self.device, obs_context_len, num_obs)
    
    def _setup_spaces(self):
        """Setup observation and action spaces for HARL interface."""
        # Get observation dimension from environment or use default
        obs_dim = getattr(self, 'num_obs', 84)  # HST observation dimension
        
        # For upper-level HH policy, we can use the same observation
        # or a subset focused on task-relevant information
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        
        # Action space: 19-dim action CORRECTIONS on top of HST baseline
        # Using smaller range [-0.5, 0.5] for stable learning
        # Output 0 = use HST's baseline action unchanged
        action_scale = self.env_args.get("action_scale", 0.5)
        self.action_scale = action_scale
        self.action_space = [
            spaces.Box(low=-action_scale, high=action_scale, shape=(self.num_dofs,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        
        # Shared observation space (same as individual observation for single agent)
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
    
    def _update_obs_with_target(self, target_jt):
        """
        Update the observation buffer to include new target joint positions.
        
        This is critical for the hierarchical control to work correctly.
        The HST observation vector contains delayed_obs_target_jt at positions 46-64
        (after base_orn_rp[2], base_ang_vel[3], commands[3], dof_pos-default[19], 
        dof_vel[19] = 2+3+3+19+19=46).
        
        Args:
            target_jt: New target joint positions, shape (n_envs, num_dofs)
        """
        # Update delayed_obs_target_jt in environment
        if hasattr(self.env, 'delayed_obs_target_jt'):
            self.env.delayed_obs_target_jt = target_jt.clone()
        
        # The HST obs_buf contains:
        # [0:2] base_orn_rp (roll, pitch)
        # [2:5] base_ang_vel (3)
        # [5:8] commands (3)
        # [8:27] dof_pos - default_dof_pos (19)
        # [27:46] dof_vel (19)
        # [46:65] delayed_obs_target_jt (19)
        # [65:84] last_actions (19)
        # Total: 84
        
        target_jt_start_idx = 46
        target_jt_end_idx = 65
        
        # Update obs_buf with new target joint positions
        if hasattr(self.env, 'obs_buf') and self.env.obs_buf is not None:
            # Normalize target_jt relative to default_dof_pos (same as HST does)
            if hasattr(self.env, 'default_dof_pos') and self.env.default_dof_pos is not None:
                default_pos = self.env.default_dof_pos
                if not isinstance(default_pos, torch.Tensor):
                    default_pos = torch.tensor(default_pos, device=self.device, dtype=torch.float32)
                if default_pos.dim() == 1:
                    default_pos = default_pos.unsqueeze(0)
                # HST stores target_jt - default_dof_pos in obs_buf
                target_offset = target_jt - default_pos
            else:
                target_offset = target_jt
            
            # Update the target joint portion in obs_buf
            self.env.obs_buf[:, target_jt_start_idx:target_jt_end_idx] = target_offset
        
        # Update obs_history_buf with new observation
        # obs_history_buf shape: (n_envs, context_len, obs_dim)
        if hasattr(self.env, 'obs_history_buf') and self.env.obs_history_buf is not None:
            if self.env.obs_history_buf.dim() == 3:
                # Update the last frame in history (most recent observation)
                if hasattr(self.env, 'obs_buf') and self.env.obs_buf is not None:
                    # Shift history and add new observation
                    self.env.obs_history_buf[:, :-1] = self.env.obs_history_buf[:, 1:].clone()
                    self.env.obs_history_buf[:, -1] = self.env.obs_buf.clone()
    
    def _load_pretrained_hst(self):
        """Load pretrained HST policy weights."""
        if self.hst_checkpoint is None:
            print("No HST checkpoint specified, using random initialization")
            return
            
        try:
            from rsl_rl.modules import ActorCriticTransformer
            
            # Try loading with weights_only=True first for security,
            # but fall back to weights_only=False for older checkpoints
            try:
                checkpoint = torch.load(
                    self.hst_checkpoint, 
                    map_location=self.device,
                    weights_only=True
                )
            except Exception:
                # Fallback for older checkpoints that may contain non-tensor objects
                print("Note: Loading checkpoint with weights_only=False (older format)")
                checkpoint = torch.load(
                    self.hst_checkpoint, 
                    map_location=self.device,
                    weights_only=False
                )
            
            # Initialize HST policy network
            self.hst_policy = ActorCriticTransformer(
                num_actor_obs=84,
                num_critic_obs=84,
                num_actions=self.num_dofs,
                obs_context_len=8
            ).to(self.device)
            
            self.hst_policy.load_state_dict(checkpoint['model_state_dict'])
            self.hst_policy.eval()
            
            print(f"Successfully loaded pretrained HST from {self.hst_checkpoint}")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained HST: {e}")
            print("Training will proceed without pretrained HST (upper-level only)")
            self.hst_policy = None
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            obs: Observations for each agent, shape (n_envs, n_agents, obs_dim)
            share_obs: Shared observations, shape (n_envs, n_agents, obs_dim)  
            available_actions: None (continuous action space)
        """
        self.current_step = 0
        
        # Reset HST environment
        # HST reset() returns (obs_history_buf, privileged_obs)
        reset_result = self.env.reset()
        
        # Handle different return formats
        if isinstance(reset_result, tuple):
            obs_history = reset_result[0]
        else:
            obs_history = reset_result
        
        # Get the last observation from history
        # obs_history shape: (n_envs, context_len, obs_dim) or (n_envs, obs_dim)
        if isinstance(obs_history, torch.Tensor):
            if obs_history.dim() == 3:
                obs = _t2n(obs_history[:, -1, :])  # Take last timestep
            else:
                obs = _t2n(obs_history)  # Already single frame
        else:
            if len(obs_history.shape) == 3:
                obs = obs_history[:, -1, :]
            else:
                obs = obs_history
        
        # Reshape for HARL interface: (n_envs, n_agents, obs_dim)
        obs = obs.reshape(self.n_envs, 1, -1)
        share_obs = obs.copy()
        
        return obs, share_obs, [None] * self.n_envs
    
    def step(self, actions):
        """
        Execute one environment step.
        
        NEW APPROACH: HH outputs action CORRECTIONS on top of HST's base actions.
        This allows HH to learn while leveraging HST's stable baseline behavior.
        
        Args:
            actions: Action corrections from HH policy, shape (n_envs, n_agents, 19)
                     These are ADDED to HST's baseline actions
            
        Returns:
            obs: Next observations, shape (n_envs, n_agents, obs_dim)
            share_obs: Shared observations, shape (n_envs, n_agents, obs_dim)
            rewards: Rewards, shape (n_envs, n_agents, 1)
            dones: Done flags, shape (n_envs, n_agents)
            infos: Additional info dicts
            available_actions: None (continuous action space)
        """
        self.current_step += 1
        
        # Convert actions to torch tensor if needed
        # actions shape: (n_envs, n_agents, 19) -> (n_envs, 19)
        if isinstance(actions, np.ndarray):
            action_corrections = torch.from_numpy(actions[:, 0, :]).float().to(self.device)
        else:
            action_corrections = actions[:, 0, :].float().to(self.device)
        
        # Get HST's baseline actions (the stable policy output)
        if hasattr(self, 'hst_policy') and self.hst_policy is not None:
            with torch.no_grad():
                # Get observation history from environment
                obs_history = self.env.obs_history_buf if hasattr(self.env, 'obs_history_buf') else None
                if obs_history is not None:
                    hst_base_actions = self.hst_policy.act_inference(obs_history)
                else:
                    hst_base_actions = torch.zeros(self.n_envs, self.num_dofs, device=self.device)
        else:
            # Without pretrained HST, use zero baseline
            hst_base_actions = torch.zeros(self.n_envs, self.num_dofs, device=self.device)
        
        # Combine HST baseline with HH corrections
        # HH learns to output corrections that improve upon HST's baseline
        final_actions = hst_base_actions + action_corrections
        
        # Step the HST environment with combined actions
        # HST step() returns (obs_history_buf, privileged_obs, rew_buf, reset_buf, extras)
        step_result = self.env.step(final_actions)
        obs_history = step_result[0]
        rewards = step_result[2]
        dones = step_result[3]
        extras = step_result[4] if len(step_result) > 4 else {}
        
        # Convert outputs to numpy
        # obs_history shape: (n_envs, context_len, obs_dim) or (n_envs, obs_dim)
        if isinstance(obs_history, torch.Tensor):
            if obs_history.dim() == 3:
                obs = _t2n(obs_history[:, -1, :])  # Take last timestep
            else:
                obs = _t2n(obs_history)
        else:
            if len(obs_history.shape) == 3:
                obs = obs_history[:, -1, :]
            else:
                obs = obs_history
            
        if isinstance(rewards, torch.Tensor):
            rewards = _t2n(rewards)
        if isinstance(dones, torch.Tensor):
            dones = _t2n(dones)
        
        # Reshape for HARL interface
        obs = obs.reshape(self.n_envs, 1, -1)
        share_obs = obs.copy()
        rewards = rewards.reshape(self.n_envs, 1, 1)
        dones = dones.reshape(self.n_envs, 1)
        
        # Create info dicts
        infos = [[{}] for _ in range(self.n_envs)]
        
        # Check for episode timeout
        if self.current_step >= self.max_episode_length:
            dones[:] = 1
            for i in range(self.n_envs):
                infos[i][0]["bad_transition"] = True
        
        return obs, share_obs, rewards, dones, infos, [None] * self.n_envs
    
    def seed(self, seed):
        """Set random seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def render(self):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            self.env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def save_video(self, filename):
        """
        Save a video of the current episode.
        
        Args:
            filename: Path to save the video
        """
        # IsaacGym video recording would be implemented here
        # This requires setting up camera and recording frames
        print(f"Video saving to {filename} - not yet implemented")


class HumanPlusHierarchicalEnv(HumanPlusEnv):
    """
    Hierarchical training environment for HH-HST integration.
    
    This environment supports:
    1. Phase 1: Training HST independently (use standard HST training)
    2. Phase 2: Training upper-level HH with frozen HST
    3. Phase 3: Joint fine-tuning of both layers
    
    Attributes:
        freeze_hst: Whether to freeze HST during training
        training_phase: Current training phase (1, 2, or 3)
    """
    
    def __init__(self, env_args):
        """
        Initialize hierarchical environment.
        
        Args:
            env_args: Environment arguments including:
                - freeze_hst: Whether to freeze HST weights
                - training_phase: Training phase (1, 2, or 3)
        """
        super().__init__(env_args)
        
        self.freeze_hst = env_args.get("freeze_hst", True)
        self.training_phase = env_args.get("training_phase", 2)
        
        if self.freeze_hst and self.hst_policy is not None:
            for param in self.hst_policy.parameters():
                param.requires_grad = False
    
    def set_training_phase(self, phase):
        """
        Set the training phase.
        
        Args:
            phase: Training phase (1=HST only, 2=HH only, 3=joint)
        """
        self.training_phase = phase
        
        if phase == 1:
            # Phase 1: Train HST only (not applicable through this wrapper)
            pass
        elif phase == 2:
            # Phase 2: Train HH with frozen HST
            self.freeze_hst = True
            if self.hst_policy is not None:
                for param in self.hst_policy.parameters():
                    param.requires_grad = False
        elif phase == 3:
            # Phase 3: Joint training
            self.freeze_hst = False
            if self.hst_policy is not None:
                for param in self.hst_policy.parameters():
                    param.requires_grad = True
