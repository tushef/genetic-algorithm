import pathlib

import ray
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt

BASE_PATH = pathlib.Path(__file__).resolve().parent
TUNER_DIR = (BASE_PATH / 'neural_architecture_search').resolve()

def train_rl(config):
    """
    Training function for Ray Tune. Trains a PPO agent on a Gym environment
    using hyperparameters and architecture specified in `config`.
    Periodically evaluates the agent and reports average reward to Ray Tune.
    """
    import niryo_gym # Needs to be called within the Ray Worker

    env = make_vec_env(
        "NedReach-v1",
        n_envs=1,
        env_kwargs={"render_mode": "none"}
    )

    # Define architecture dynamically from Ray config
    policy_kwargs = {
        "net_arch": [int(config["layer1"]), int(config["layer2"]), int(config["layer3"])],
    }

    # model = PPO(
    #     policy="MlpPolicy",             # Use MLP for policy & value networks
    #     env=env,
    #     learning_rate=config["lr"],     # Learning rate to tune
    #     gamma=config["gamma"],          # Discount factor to tune
    #     ent_coef=config["ent_coef"],    # Entropy regularization to tune
    #     n_steps=config["n_steps"],      # Rollout buffer size before update
    #     batch_size=config["batch_size"],# Mini-batch size for optimization
    #     policy_kwargs=policy_kwargs,    # Custom network architecture
    #     verbose=0
    # )

    print(f"Current architecture: {config['layer1']}-{config['layer2']}-{config['layer3']}")

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["lr"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        policy_kwargs=policy_kwargs,
        learning_starts=config['learning_starts'],
        gradient_steps=config['gradient_steps'],
        verbose=0
    )

    total_timesteps = 200000 # Recommended for Reach
    eval_freq = 1000
    current_timestep = 0

    for i in range(0, total_timesteps, eval_freq):
        # Train for a chunk of steps
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)

        # Evaluate the current policy
        rewards = []
        for _ in range(5):
            obs = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]

            rewards.append(total_reward)
        current_timestep += eval_freq
        # Report average reward across episodes to Ray Tune
        tune.report({"mean_reward": np.mean(rewards),
                     "global_step": current_timestep,
                     "training_iteration": i // eval_freq})

if __name__ == "__main__":
    ray.init(num_cpus=4, _temp_dir=str(TUNER_DIR))

    perturbation_interval = 10

    #
    # pbt = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     perturbation_interval=2,
    #     hyperparam_mutations={
    #         "lr": tune.choice([5e-4, 1e-4]),
    #         "layer1": tune.choice([64, 128, 256]),
    #         "layer2": tune.choice([64, 128, 256]),
    #         "layer3": tune.choice([64, 128, 256]),
    #     }
    # )
    #
    # search_space = {
    #     "lr": tune.choice([5e-4, 1e-4]),
    #     "gamma": tune.choice([0.95]),
    #     # "ent_coef": tune.uniform(0.001, 0.1),  # or use 'auto'
    #     "buffer_size": tune.choice([25000]),
    #     "batch_size": tune.choice([512]),  # larger batch sizes are common in SAC
    #     "layer1": tune.choice([64, 128, 256]),
    #     "layer2": tune.choice([64, 128, 256]),
    #     "layer3": tune.choice([64, 128, 256]),
    #     # "train_freq": tune.choice([8]),  # number of steps before each training step
    #     "gradient_steps": tune.choice([10]),  # number of gradient steps per update
    #     "learning_starts": tune.choice([5000])
    # }
    #
    # results_grid = tune.run(
    #     train_rl,
    #     config=search_space,
    #     scheduler=pbt,
    #     num_samples=10,  # Number of parallel trials
    #     resources_per_trial={"cpu": 1},
    #     metric="mean_reward",
    #     mode="max",
    #     storage_path=TUNER_DIR
    # )

    ### Second Iteration ###

    # Define the scheduler
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_mutations={
            "lr": tune.choice([5e-4, 1e-4]),
            "layer1": tune.choice([64, 128, 256]),
            "layer2": tune.choice([64, 128, 256]),
            "layer3": tune.choice([64, 128, 256]),
        }
    )

    # Define the search space
    search_space = {
        "lr": tune.choice([5e-4, 1e-4]),
        "gamma": tune.choice([0.95]),
        "buffer_size": tune.choice([25000]),
        "batch_size": tune.choice([512]),
        "layer1": tune.choice([64, 128, 256]),
        "layer2": tune.choice([64, 128, 256]),
        "layer3": tune.choice([64, 128, 256]),
        "gradient_steps": tune.choice([10]),
        "learning_starts": tune.choice([5000])
    }

    trainable_with_resources = tune.with_resources(train_rl, {"cpu": 1})

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=10,
            metric="mean_reward",
            mode="max"
        ),
        run_config=tune.RunConfig(
            storage_path=str(TUNER_DIR),
            name="pbt_sac_grasping"
        )
    )

    results_grid = tuner.fit()

    # trial_dfs = results_grid.trial_dataframes
    #
    # # Get the best trial result
    # best_result = results_grid.get_best_trial(metric="mean_reward", mode="max")
    # best_trial_id = best_result.trial_id
    # print(f"Best trial: {best_trial_id}")
    #
    # plt.figure(figsize=(10, 6))
    # for trial_id, df in trial_dfs.items():
    #     plt.plot(df.index, df["mean_reward"], label=trial_id)
    # plt.xlabel("Evaluation Step")
    # plt.ylabel("Mean Reward")
    # plt.title("All Trials: Mean Reward Progression")
    # plt.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # best_df = trial_dfs[best_trial_id]
    #
    # plt.figure(figsize=(8, 5))
    # plt.plot(best_df.index, best_df["mean_reward"], marker="o", color="blue")
    # plt.xlabel("Evaluation Step")
    # plt.ylabel("Mean Reward")
    # plt.title(f"Best Trial ({best_trial_id}): Mean Reward Progression")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # print(results_grid.trial_dataframes)
    #
    # # Print `path` where checkpoints are stored
    # print('Best result path:', best_result.path)
    #
    # # Print the best trial `config` reported at the last iteration
    # # NOTE: This config is just what the trial ended up with at the last iteration.
    # # See the next section for replaying the entire history of configs.
    # print("Best final iteration hyperparameter config:\n", best_result.config)

    ### Second Version Plotting ###

    # Get the best trial result
    best_result = results_grid.get_best_result(metric="mean_reward", mode="max")

    # Print `path` where checkpoints are stored
    print('Best result path:', best_result.path)

    # Print the best trial `config` reported at the last iteration
    # NOTE: This config is just what the trial ended up with at the last iteration.
    # See the next section for replaying the entire history of configs.
    print("Best final iteration hyperparameter config:\n", best_result.config)

    # Plot the learning curve for the best trial
    df = best_result.metrics_dataframe
    # Deduplicate, since PBT might introduce duplicate data
    df = df.drop_duplicates(subset="training_iteration", keep="last")
    df.plot("training_iteration", "mean_reward")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()