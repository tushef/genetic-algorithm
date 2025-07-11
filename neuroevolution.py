import copy
import glob
import json
import os
import uuid
from datetime import datetime

import numpy as np
import niryo_gym
import gymnasium as gym
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

from scipy.spatial.distance import pdist

""" Working Prototype on NedReach """

def evaluate_success_rate(model, episodes=30, render=False, env_name='CartPole-v1', success_threshold=500):
    env = gym.make(env_name)
    success_count = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if render:
                env.render()

            action, _ = model.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = action.detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if terminated:
                success_count += 1

    env.close()
    return success_count / episodes

def evaluate_policy(model, episodes=30, render=False, env_name='CartPole-v1'):
    env = gym.make(env_name)
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            # select action from dist
            action, _ = model.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = action.detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    env.close()
    return total_reward / episodes


def fitness_fn_v1(model, episodes=30, render=False, penalty_weight=1.0, env_name='CartPole-v1'):
    env = gym.make(env_name)
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_episode_reward = 0.0

        while not done:
            if render:
                env.render()
            action, _ = model.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = action.detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_episode_reward += reward

        episode_rewards.append(total_episode_reward)

    env.close()

    rewards = np.array(episode_rewards)
    mean_reward = rewards.mean()
    std_reward = rewards.std()

    # Penalize high variance
    adjusted_fitness = mean_reward - penalty_weight * std_reward

    return adjusted_fitness

def compute_cvar(rewards, alpha=0.2):
    """
    Computes the Conditional Value at Risk (CVaR) at confidence level alpha.
    Lower CVaR → worse tail performance.
    """
    sorted_rewards = np.sort(rewards)
    cutoff = max(1, int(np.ceil(alpha * len(sorted_rewards))))
    worst_rewards = sorted_rewards[:cutoff]
    return np.mean(worst_rewards)

def evaluate_policy_inverse_cvar(model, episodes=30, alpha=0.2, epsilon=1e-6, render=False, env_name='CartPole-v1'):
    """
    Evaluate a model using a fitness score that penalizes std with the inverse of CVaR.
    """
    env = gym.make(env_name)
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if render:
                env.render()
            action, _ = model.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = action.detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()

    rewards = np.array(episode_rewards)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    cvar = compute_cvar(rewards, alpha=alpha)

    # CVaR inversion
    penalty_weight = 1.0 / (cvar + epsilon)
    if penalty_weight < 0.0:
        penalty_weight = -penalty_weight
    fitness = mean_reward - penalty_weight * std_reward

    return fitness

class DynamicNet(nn.Module):
    def __init__(self, hidden_dims, activations=None):
        super(DynamicNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations or ['relu'] * (len(hidden_dims) - 2) + ['identity']
        self.backbone = self._build_network()

    def _get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'identity': nn.Identity()
        }.get(name.lower(), nn.ReLU())

    def _build_network(self):
        layers = []
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            if i < len(self.activations):
                layers.append(self._get_activation(self.activations[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.backbone(x)

    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_weights(self, flat_weights):
        with torch.no_grad():
            offset = 0
            for param in self.parameters():
                size = param.numel()
                param.copy_(flat_weights[offset:offset + size].view_as(param))
                offset += size

class Actor(DynamicNet):
    """
    Soft Actor Critic Policy Network
    """
    def __init__(self, hidden_dims, output_dim, max_action, activations=None):
        super(Actor, self).__init__(hidden_dims, activations)
        self.max_action = max_action
        self.mean = nn.Linear(hidden_dims[-1], output_dim)
        self.log_std = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist

    def sample(self, state):
        dist = self.forward(state)
        x_t = dist.rsample()
        action = torch.tanh(x_t) * self.max_action
        log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob

class EarlyStoppingConfig:
    def __init__(self, patience=10, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def check(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class SelectionStrategy:
    def __init__(self, method='top_k', **kwargs):
        self.method = method
        self.kwargs = kwargs

    def select(self, population, fitnesses):
        if self.method == 'top_k':
            return self._top_k(population, fitnesses, self.kwargs.get('k', len(population) // 2))
        elif self.method == 'tournament':
            return self._tournament(population, fitnesses, self.kwargs.get('tournament_size', 3))
        elif self.method == 'roulette':
            return self._roulette(population, fitnesses)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

    def _top_k(self, population, fitnesses, k):
        sorted_indices = np.argsort(fitnesses)[-k:]
        return [population[i] for i in sorted_indices]

    def _tournament(self, population, fitnesses, tournament_size):
        selected = []
        for _ in range(len(population) // 2):
            contenders = np.random.choice(len(population), tournament_size, replace=False)
            winner = max(contenders, key=lambda i: fitnesses[i])
            selected.append(population[winner])
        return selected

    def _roulette(self, population, fitnesses):
        fitness_sum = np.sum(fitnesses)
        probs = fitnesses / fitness_sum
        selected_indices = np.random.choice(len(population), len(population) // 2, p=probs)
        return [population[i] for i in selected_indices]

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size,
        net_config,
        fitness_fn,
        crossover_strategy='mixed',  # 'mean', 'min', 'max', 'random', or 'mixed'
        mutation_rate=0.1,
        perturbation_interval=5,
        selection_strategy=None,
        early_stopping=None,
        env_name='CartPole-v1',
        verbose=True,
    ):
        self.pop_size = pop_size
        self.net_config = net_config
        self.fitness_fn = fitness_fn
        self.crossover_strategy = crossover_strategy
        self.mutation_rate = mutation_rate
        self.perturbation_interval = perturbation_interval
        self.selection_strategy = selection_strategy or SelectionStrategy('top_k')
        self.early_stopping = early_stopping or EarlyStoppingConfig(patience=5, min_delta=1e-2)
        self.env_name = env_name
        self.verbose = verbose

        # Initialize random population
        self._load_population_from_checkpoints(checkpoints_dir="neuroevolution/results")
        # self.population = [DynamicNet(**net_config) for _ in range(pop_size)]
        self.elite_pool = []  # Holds tuples of (model, fitness)
        self.elite_pool_size = 1
        self.checkpoint_elites = True
        self.checkpoint_interval = 5
        self.elite_changed_counter = 0 # internal metrics
        self._init_checkpoint_dir()

    def _load_population_from_checkpoints(self, checkpoints_dir):
        """
        Initializes the population from all checkpoint files found recursively in the given directory.
        Fills remaining population with random models if not enough checkpoints exist.
        """
        checkpoint_paths = glob.glob(os.path.join(checkpoints_dir, "**", "*.pth"), recursive=True)
        print(f"Found {len(checkpoint_paths)} checkpoint(s).")

        self.population = []  # Just to be sure

        for path in checkpoint_paths:
            model = Actor(**self.net_config)
            state_dict = torch.load(path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            self.population.append(model)

            if len(self.population) >= self.pop_size:
                break  # Stop if we've reached the desired population size

        while len(self.population) < self.pop_size:
            model = Actor(**self.net_config)
            self.population.append(model)

    def _get_weights(self, model):
        return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    def _set_weights(self, model, flat_weights):
        model.set_weights(flat_weights)

    def _save_elite_checkpoint(self, model, fitness, generation, elite_id):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"elite{elite_id}_gen{generation}_fitness{fitness:.2f}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(model.state_dict(), path)

    def _init_checkpoint_dir(self, base_dir="neuroevolution"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]  # Short unique ID
        folder_name = f"run_{timestamp}_{unique_id}"
        path = os.path.join(base_dir, folder_name)
        os.makedirs(path, exist_ok=True)
        self.checkpoint_dir = path
        self._save_run_config()

    def _save_run_config(self):
        # For Reproducability
        config = {
            "population_size": self.pop_size,
            "elite_pool_size": self.elite_pool_size,
            "mutation_rate": self.mutation_rate,
            "crossover_strategy": self.crossover_strategy,
            "perturbation_interval": self.perturbation_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "selection_method": self.selection_strategy.method,
            "fitness_function": self.fitness_fn.__name__,
            "early_stopping": {
                "patience": getattr(self.early_stopping, 'patience', None),
                "delta": getattr(self.early_stopping, 'min_delta', None)
            },
            "net_config": self.net_config,
            "seed": getattr(self, "seed", None)
        }

        config_path = os.path.join(self.checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def _update_elite_pool(self, population, fitnesses, generation):
        for model, fitness in zip(population, fitnesses):
            if len(self.elite_pool) < self.elite_pool_size:
                self.elite_pool.append((copy.deepcopy(model), fitness))
            else:
                # Replace worst elite if better
                worst_idx = np.argmin([f for _, f in self.elite_pool])
                if fitness > self.elite_pool[worst_idx][1]:
                    self.elite_pool[worst_idx] = (copy.deepcopy(model), fitness)
                    self.elite_changed_counter += 1

        # Checkpointing
        if self.checkpoint_elites and generation > 0 and generation % self.checkpoint_interval == 0:
            for i, (model, fitness) in enumerate(self.elite_pool):
                self._save_elite_checkpoint(model, fitness, generation, elite_id=i)

    def _get_elite(self):
        elite = sorted(self.elite_pool, key=lambda x: x[1])
        return elite

    def _crossover(self, w1, w2):
        if self.crossover_strategy == 'random':
            mask = torch.rand_like(w1) > 0.5
            return torch.where(mask, w1, w2)
        elif self.crossover_strategy == 'mean':
            return (w1 + w2) / 2
        elif self.crossover_strategy == 'min':
            return torch.min(w1, w2)
        elif self.crossover_strategy == 'max':
            return torch.max(w1, w2)
        elif self.crossover_strategy == 'mixed':
            ops = [torch.min, torch.max, lambda a, b: (a + b) / 2, self._midpoint_crossover_op]
            op = np.random.choice(ops)
            return op(w1, w2)
        else:
            raise ValueError(f"Unknown crossover strategy: {self.crossover_strategy}")

    def _midpoint_crossover_op(self, weights1, weights2):
        """
        Custom 1-point crossover: splits at the middle and combines halves from two parents.
        """
        assert len(weights1) == len(weights2), "Parents must have the same number of weights."

        crossover_point = len(weights1) // 2

        # Combine first half from weights1 and second half from weights2
        child_weights = torch.cat((weights1[:crossover_point], weights2[crossover_point:]), dim=0)

        return child_weights

    def _mutate(self, weights):
        noise = torch.randn_like(weights) * self.mutation_rate
        return weights + noise

    def evolve(self, max_generations=50):
        log_data = []

        for gen in range(max_generations):
            fitnesses = np.array([self.fitness_fn(ind, env_name=self.env_name) for ind in self.population])
            top_score = np.max(fitnesses)
            std_score = np.std(fitnesses)
            min_score = np.min(fitnesses)
            noise_range = np.max(fitnesses) - np.min(fitnesses)
            score_iqr = np.percentile(fitnesses, 75) - np.percentile(fitnesses, 25)

            weights = [self._get_weights(ind) for ind in self.population]
            flat_weights = np.array([w.flatten() for w in weights])
            param_std = np.std(flat_weights)  # Total std across all parameters

            # Average pairwise L2 distance
            if len(flat_weights) > 1:
                pairwise_dists = pdist(flat_weights, metric='euclidean')
                param_l2_diversity = np.mean(pairwise_dists)
            else:
                param_l2_diversity = 0.0

            # Log generation data
            log_data.append({
                "generation": gen,
                # Core Fitness Scores
                "top_score": top_score,
                "min_score": min_score,
                "std_score": std_score,
                "noise_range": noise_range,
                "score_iqr": score_iqr,
                # Population Diversity Metrics
                "param_std": param_std,
                "param_l2_diversity": param_l2_diversity,
                "elite_changed": self.elite_changed_counter
            })

            # Update elite pool
            self._update_elite_pool(self.population, fitnesses, gen)

            if self.verbose:
                print(f"Gen {gen} | Top: {top_score:.4f} | Std: {np.std(fitnesses):.4f} | IQR: {score_iqr:.4f} | Diversity: {param_l2_diversity:.4f} | Elite Changed: {self.elite_changed_counter}")
            # Early stopping
            if self.early_stopping and self.early_stopping.check(top_score):
                print("Early stopping triggered.")
                break

            # Selection
            selected = self.selection_strategy.select(self.population, fitnesses)

            new_population = []

            # Preserve the elites into the new generation
            for model, _ in self.elite_pool:
                new_population.append(copy.deepcopy(model))

            # Reproduction
            while len(new_population) < self.pop_size:
                p1, p2 = np.random.choice(selected, 2, replace=False)
                w1 = self._get_weights(p1)
                w2 = self._get_weights(p2)
                child_weights = self._crossover(w1, w2)

                if gen % self.perturbation_interval == 0:
                    # We do the mutation only to child weights not to elites
                    child_weights = self._mutate(child_weights)

                child = Actor(**self.net_config)
                self._set_weights(child, child_weights)
                new_population.append(child)

            self.population = new_population

        log_df = pd.DataFrame(log_data)
        log_df.to_csv("evolution_log.csv", index=False)


if __name__ == "__main__":
    early_stopping = EarlyStoppingConfig(patience=25, min_delta=0.01)
    selection = SelectionStrategy(method='tournament', tournament_size=3)

    env = gym.make('NedReach-v1')
    obs, _ = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    ga = GeneticAlgorithm(
        pop_size=50,
        net_config={'hidden_dims': [state_dim, 256, 128, 64],
                    'activations': ['relu', 'relu', 'relu'],
                    'max_action': max_action,
                    'output_dim': action_dim
                    },
        fitness_fn=fitness_fn_v1,
        crossover_strategy='mixed',
        mutation_rate=0.1,
        perturbation_interval=10,
        selection_strategy=selection,
        early_stopping=early_stopping,
        env_name='NedReach-v1',
        verbose=True
    )

    ga.evolve(max_generations=50)

    # Final Testing
    actor = Actor(hidden_dims=[state_dim, 256, 128, 64], activations=['relu', 'relu', 'relu'],
                  output_dim=action_dim, max_action=max_action)

    # Access the model from elite_pool contains (model, fitness) tuples
    best_model = ga.elite_pool[0][0] if isinstance(ga.elite_pool[0], tuple) else ga.elite_pool[0]
    actor.load_state_dict(best_model.state_dict())
    fitness_score = evaluate_policy(actor, env_name='NedReach-v1')
    print(fitness_score)

