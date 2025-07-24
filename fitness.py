import gym
import torch
import numpy as np

class FitnessFunction:
    def __init__(self, model, env_name='CartPole-v1', episodes=30, render=False):
        self.model = model
        self.env_name = env_name
        self.episodes = episodes
        self.render = render

    def _run_episodes(self):
        env = gym.make(self.env_name)
        episode_rewards = []
        success_count = 0
        info_flags = []

        for _ in range(self.episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            episode_info = {}

            while not done:
                if self.render:
                    env.render()

                action, _ = self.model.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = action.detach().cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                episode_info.update(info)

            episode_rewards.append(total_reward)
            info_flags.append(episode_info)

        env.close()
        return np.array(episode_rewards), info_flags

    def _evaluate_success_rate(self):
        _, info_flags = self._run_episodes()
        success_count = sum(info.get("is_success", 0) for info in info_flags)
        return success_count / self.episodes

    def _evaluate_policy_mean(self):
        rewards, _ = self._run_episodes()
        return rewards.mean()

    def _evaluate_penalized_std(self, penalty_weight=1.0):
        rewards, _ = self._run_episodes()
        return rewards.mean() - penalty_weight * rewards.std()

    def _compute_cvar(self, rewards, alpha=0.2):
        sorted_rewards = np.sort(rewards)
        cutoff = max(1, int(np.ceil(alpha * len(sorted_rewards))))
        return np.mean(sorted_rewards[:cutoff])

    def _evaluate_policy_cvar(self, quantile=0.2, alpha=1.0, beta=0.5):
        rewards, _ = self._run_episodes()
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        cvar = self._compute_cvar(rewards, alpha=quantile)
        return mean_reward - alpha * std_reward - beta * abs(cvar)

    def evaluate(self, method='mean', **kwargs):
        if method == 'success_rate':
            return self._evaluate_success_rate()
        elif method == 'mean':
            return self._evaluate_policy_mean()
        elif method == 'penalized_std':
            return self._evaluate_penalized_std(**kwargs)
        elif method == 'cvar':
            return self._evaluate_policy_cvar(**kwargs)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")