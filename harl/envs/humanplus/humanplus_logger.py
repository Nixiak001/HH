"""Logger for HumanPlus environment."""
import time
import numpy as np
from harl.common.base_logger import BaseLogger


class HumanPlusLogger(BaseLogger):
    """Logger for HumanPlus HST environment."""
    
    def get_task_name(self):
        """Get the task name for logging."""
        return self.env_args.get("task", "h1_walking")
    
    def init(self, episodes):
        """Initialize the logger."""
        super().init(episodes)
        self.episode_rewards = []
        self.episode_lengths = []
        self.target_tracking_errors = []
    
    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, "
            "total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.log_file.write(
                ",".join(map(str, [self.total_num_steps, aver_episode_rewards])) + "\n"
            )
            self.log_file.flush()
            self.done_episodes_rewards = []
    
    def eval_log(self, eval_episode):
        """Log evaluation results."""
        super().eval_log(eval_episode)
        
        # Log additional HumanPlus-specific metrics
        if len(self.target_tracking_errors) > 0:
            avg_tracking_error = np.mean(self.target_tracking_errors)
            self.writter.add_scalar(
                "eval/target_tracking_error",
                avg_tracking_error,
                self.total_num_steps,
            )
            self.target_tracking_errors = []
    
    def per_step(self, data):
        """Process per-step data for logging."""
        super().per_step(data)
        
        # Extract additional metrics from infos if available
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        
        # Log target tracking error if available in infos
        for info_list in infos:
            for info in info_list:
                if isinstance(info, dict) and "target_tracking_error" in info:
                    self.target_tracking_errors.append(info["target_tracking_error"])
