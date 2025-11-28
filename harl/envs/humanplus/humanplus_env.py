"""
HumanPlus HST Environment Wrapper for HARL.

This module provides the integration between HH (HARL) upper-level policy 
and HumanPlus HST (Humanoid Shadowing Transformer) lower-level controller.

Architecture:
    Upper Layer (HH/HARL): Outputs 19-dim target joint positions
    Lower Layer (HST): Receives target poses + proprioception, outputs joint torques
    Simulation: IsaacGym physics simulation

The upper-level HH policy learns to output target joint positions that the 
pre-trained HST controller can track to accomplish walking tasks.
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
    
    The key modification is that instead of loading target joint trajectories
    from pre-recorded files, the target joints come from the upper-level HARL
    policy output.
    
    Attributes:
        n_envs: Number of parallel environments
        n_agents: Number of agents (1 for single humanoid)
        num_dofs: Number of degrees of freedom (19 for H1 robot)
        observation_space: Observation space for each agent
        action_space: Action space for each agent (19-dim target joint positions)
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
            
            # Create H1 environment
            env = H1(
                cfg=H1RoughCfg,
                sim_params=sim_params,
                physics_engine=gymapi.SIM_PHYSX,
                sim_device=self.device,
                headless=self.headless
            )
            
            return env
            
        except ImportError as e:
            print(f"Warning: Could not import humanplus HST environment: {e}")
            print("Creating mock environment for development/testing...")
            return self._create_mock_env()
    
    def _create_mock_env(self):
        """
        Create a mock environment for testing without humanplus installation.
        
        Returns:
            Mock environment object
        """
        class MockHSTEnv:
            def __init__(self, num_envs, num_dofs, device):
                self.num_envs = num_envs
                self.num_dofs = num_dofs
                self.device = device
                self.num_obs = 84  # Same as HST observation dimension
                
                # Mock state tensors
                self.obs_buf = torch.zeros(num_envs, self.num_obs, device=device)
                self.rew_buf = torch.zeros(num_envs, device=device)
                self.reset_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
                self.dof_pos = torch.zeros(num_envs, num_dofs, device=device)
                self.default_dof_pos = torch.zeros(num_dofs, device=device)
                
            def reset(self):
                self.obs_buf = torch.randn(self.num_envs, self.num_obs, device=self.device) * 0.1
                obs_history = self.obs_buf.unsqueeze(1).repeat(1, 8, 1)  # context_len=8
                return obs_history, None
            
            def step(self, actions):
                # Simulate one step
                self.obs_buf = torch.randn(self.num_envs, self.num_obs, device=self.device) * 0.1
                obs_history = self.obs_buf.unsqueeze(1).repeat(1, 8, 1)
                self.rew_buf = torch.ones(self.num_envs, device=self.device) * 0.1
                self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                extras = {}
                return obs_history, None, self.rew_buf, self.reset_buf, extras
            
            def set_target_jt(self, target_jt):
                """Set the target joint positions from upper-level policy."""
                self.target_jt = target_jt
                
            def render(self):
                pass
        
        return MockHSTEnv(self.n_envs, self.num_dofs, self.device)
    
    def _setup_spaces(self):
        """Setup observation and action spaces for HARL interface."""
        # Observation space: HST observation (84 dim) 
        # Components: base_orn_rp(2) + ang_vel(3) + commands(3) + dof_pos(19) + dof_vel(19) + actions(19) + target_jt(19)
        obs_dim = 84
        
        # For upper-level HH policy, we can use the same observation
        # or a subset focused on task-relevant information
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        
        # Action space: 19-dim target joint positions
        # These will be passed to HST as target_jt
        self.action_space = [
            spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_dofs,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        
        # Shared observation space (same as individual observation for single agent)
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
    
    def _load_pretrained_hst(self):
        """Load pretrained HST policy weights."""
        if self.hst_checkpoint is None:
            print("No HST checkpoint specified, using random initialization")
            return
            
        try:
            from rsl_rl.modules import ActorCriticTransformer
            
            # Load the pretrained HST policy with weights_only for security
            checkpoint = torch.load(
                self.hst_checkpoint, 
                map_location=self.device,
                weights_only=True
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
            
            print(f"Loaded pretrained HST from {self.hst_checkpoint}")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained HST: {e}")
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
        obs_history, _ = self.env.reset()
        
        # Get the last observation from history
        if isinstance(obs_history, torch.Tensor):
            obs = _t2n(obs_history[:, -1, :])  # Take last timestep
        else:
            obs = obs_history[:, -1, :]
        
        # Reshape for HARL interface: (n_envs, n_agents, obs_dim)
        obs = obs.reshape(self.n_envs, 1, -1)
        share_obs = obs.copy()
        
        return obs, share_obs, [None] * self.n_envs
    
    def step(self, actions):
        """
        Execute one environment step.
        
        The actions from the upper-level HH policy are interpreted as target
        joint positions and passed to the HST controller.
        
        Args:
            actions: Target joint positions from HH policy, shape (n_envs, n_agents, 19)
            
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
            target_jt = torch.from_numpy(actions[:, 0, :]).float().to(self.device)
        else:
            target_jt = actions[:, 0, :].float().to(self.device)
        
        # Set target joint positions in HST environment
        if hasattr(self.env, 'set_target_jt'):
            self.env.set_target_jt(target_jt)
        elif hasattr(self.env, 'target_jt'):
            self.env.target_jt = target_jt
        
        # If using pretrained HST, get HST actions based on target poses
        if hasattr(self, 'hst_policy') and self.hst_policy is not None:
            with torch.no_grad():
                # Get observation history from environment
                obs_history = self.env.obs_history_buf if hasattr(self.env, 'obs_history_buf') else None
                if obs_history is not None:
                    hst_actions = self.hst_policy.act_inference(obs_history)
                else:
                    hst_actions = torch.zeros(self.n_envs, self.num_dofs, device=self.device)
        else:
            # Without pretrained HST, use target_jt directly as action offset
            # Get default_dof_pos as tensor, or use zeros if not available
            if hasattr(self.env, 'default_dof_pos') and self.env.default_dof_pos is not None:
                default_pos = self.env.default_dof_pos
                if not isinstance(default_pos, torch.Tensor):
                    default_pos = torch.tensor(default_pos, device=self.device, dtype=torch.float32)
            else:
                default_pos = torch.zeros(self.num_dofs, device=self.device, dtype=torch.float32)
            hst_actions = target_jt - default_pos
        
        # Step the HST environment with HST actions
        obs_history, _, rewards, dones, extras = self.env.step(hst_actions)
        
        # Convert outputs to numpy
        if isinstance(obs_history, torch.Tensor):
            obs = _t2n(obs_history[:, -1, :])
        else:
            obs = obs_history[:, -1, :]
            
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
