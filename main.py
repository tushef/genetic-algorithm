import niryo_gym
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

def bootstrap_ci(data, num_samples=1000, ci=95):
    """
    Bootstrap confidence interval of the mean along axis 0.
    :param data: (n_episodes, n_timesteps) array
    :param num_samples: Number of bootstrap resamples
    :param ci: Confidence level (e.g. 95)
    :return: lower_bounds, upper_bounds
    """
    n_episodes, n_timesteps = data.shape
    boot_means = np.empty((num_samples, n_timesteps))

    for i in range(num_samples):
        sample_indices = np.random.choice(n_episodes, size=n_episodes, replace=True)
        sampled_data = data[sample_indices, :]
        boot_means[i] = sampled_data.mean(axis=0)

    lower = np.percentile(boot_means, (100 - ci) / 2, axis=0)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)
    return lower, upper

if __name__ == "__main__":

    env_id = "NedReach-v1"

    # === Training Setup ===
    baseline_env_wrapper = make_vec_env(
        env_id,
        n_envs=1,
        # env_kwargs={"render_mode": "human"}
    )

    print(baseline_env_wrapper.observation_space.shape)
    print(baseline_env_wrapper.action_space.shape)
    print(baseline_env_wrapper.action_space.high)
    print(baseline_env_wrapper.action_space.low)

    # model = PPO("MlpPolicy", baseline_env_wrapper, verbose=1)
    # model.learn(total_timesteps=50000)
    # model.save("ppo_robotic_grasping")

    policy_kwargs = {
        "net_arch": [256, 128, 64],
    }

    # policy_kwargs = {
    #     "net_arch": [64, 256, 64],
    # }

    # policy_kwargs = {
    #     "net_arch": [256, 128, 128],
    # }

    # policy_kwargs = {
    #     "net_arch": [256, 256, 256],
    # }

    model = SAC("MlpPolicy", baseline_env_wrapper,
                learning_rate=0.0001,
                gamma=0.95,
                buffer_size=25000,
                batch_size=512,
                policy_kwargs=policy_kwargs,
                learning_starts=5000,
                gradient_steps=50,
                train_freq=50,
                verbose=1
                )
    model.learn(total_timesteps=200000, log_interval=10)
    model.save("sac_reach")

    # model = SAC("MlpPolicy", baseline_env_wrapper,
    #             learning_rate=0.0001,
    #             gamma=0.95,
    #             buffer_size=250000,
    #             batch_size=512,
    #             policy_kwargs=policy_kwargs,
    #             learning_starts=20000,
    #             gradient_steps=100,
    #             train_freq=(1, "episode"),
    #             verbose=1
    #             )
    # model.learn(total_timesteps=10000000, log_interval=10)
    # model.save("sac_push")

    # === Evaluation ===
    n_eval_episodes = 10
    max_steps = 100
    all_rewards = []

    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        env_kwargs={"render_mode": "human"}
    )

    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        step_rewards = []
        terminated, truncated = False, False
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            action, _ = model.predict(obs)
            obs, reward, done, info = eval_env.step(action)
            eval_env.render()
            step_rewards.append(reward[0])  # unwrap VecEnv reward
            terminated, truncated = done[0], info[0].get("truncated", False)
            steps += 1

        # Pad reward list to ensure all episodes are same length
        while len(step_rewards) < max_steps:
            step_rewards.append(0.0)

        all_rewards.append(step_rewards)

    reward_matrix = np.array(all_rewards)  # (n_eval_episodes, max_steps)
    mean_rewards = reward_matrix.mean(axis=0)
    lower, upper = bootstrap_ci(reward_matrix, num_samples=1000, ci=95)

    # === Plotting ===
    x = np.arange(max_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_rewards, label="Mean Step Reward")
    plt.fill_between(x, lower, upper, alpha=0.3, label="95% Bootstrap CI")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Step-wise Reward over {n_eval_episodes} Evaluation Episodes (Bootstrapped CI)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
