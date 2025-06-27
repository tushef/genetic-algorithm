import copy
import json
import os
import uuid
from datetime import datetime

import numpy as np
import niryo_gym
import gymnasium as gym
import torch
import torch.nn as nn

""" Working Prototype on CartPole """

def evaluate_policy(model, episodes=30, render=False):
    env = gym.make('CartPole-v1')
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            logits = model(obs)
            action = torch.argmax(logits).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    env.close()
    return total_reward / episodes


def fitness_fn_v1(model, episodes=30, render=False, penalty_weight=1.0):
    env = gym.make('CartPole-v1')
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_episode_reward = 0.0

        while not done:
            if render:
                env.render()
            logits = model(obs)
            action = torch.argmax(logits).item()
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

def evaluate_policy_inverse_cvar(model, episodes=30, alpha=0.2, epsilon=1e-6, render=False):
    """
    Evaluate a model using a fitness score that penalizes std with the inverse of CVaR.
    """
    env = gym.make('CartPole-v1')
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if render:
                env.render()
            logits = model(obs)
            action = torch.argmax(logits).item()
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
    fitness = mean_reward - penalty_weight * std_reward

    return fitness

class DynamicNet(nn.Module):
    def __init__(self, layer_dims, activations=None):
        super(DynamicNet, self).__init__()
        self.layer_dims = layer_dims
        self.activations = activations or ['relu'] * (len(layer_dims) - 2) + ['identity']
        self.model = self._build_network()

    def _get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'identity': nn.Identity()
        }.get(name.lower(), nn.ReLU())

    def _build_network(self):
        layers = []
        for i in range(len(self.layer_dims) - 1):
            layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.activations):
                layers.append(self._get_activation(self.activations[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.model(x)

    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_weights(self, flat_weights):
        with torch.no_grad():
            offset = 0
            for param in self.parameters():
                size = param.numel()
                param.copy_(flat_weights[offset:offset + size].view_as(param))
                offset += size

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
        verbose=True,
    ):
        self.pop_size = pop_size
        self.net_config = net_config
        self.fitness_fn = fitness_fn
        self.crossover_strategy = crossover_strategy
        self.mutation_rate = mutation_rate
        self.perturbation_interval = perturbation_interval
        self.selection_strategy = selection_strategy or SelectionStrategy('top_k')
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Initialize random population
        self.population = [DynamicNet(**net_config) for _ in range(pop_size)]
        self.elite_pool = []  # Holds tuples of (model, fitness)
        self.elite_pool_size = 1
        self.checkpoint_elites = True
        self.checkpoint_interval = 5
        self._init_checkpoint_dir()

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
            ops = [torch.min, torch.max, lambda a, b: (a + b) / 2]
            op = np.random.choice(ops)
            return op(w1, w2)
        else:
            raise ValueError(f"Unknown crossover strategy: {self.crossover_strategy}")

    def _mutate(self, weights):
        noise = torch.randn_like(weights) * self.mutation_rate
        return weights + noise

    def evolve(self, max_generations=50):
        for gen in range(max_generations):
            fitnesses = np.array([self.fitness_fn(ind) for ind in self.population])
            top_score = np.max(fitnesses)

            # Update elite pool
            self._update_elite_pool(self.population, fitnesses, gen)

            if self.verbose:
                print(f"Generation {gen} — Best Fitness: {top_score:.4f}")

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

                child = DynamicNet(**self.net_config)
                self._set_weights(child, child_weights)
                new_population.append(child)

            self.population = new_population


if __name__ == "__main__":
    early_stopping = EarlyStoppingConfig(patience=5, min_delta=1e-2)
    selection = SelectionStrategy(method='tournament', tournament_size=3)

    ga = GeneticAlgorithm(
        pop_size=20,
        net_config={'layer_dims': [4, 64, 64, 2], 'activations': ['relu', 'relu', 'identity']},
        fitness_fn=evaluate_policy_inverse_cvar,
        crossover_strategy='mixed',
        mutation_rate=0.1,
        perturbation_interval=5,
        selection_strategy=selection,
        early_stopping=early_stopping,
        verbose=True
    )

    ga.evolve(max_generations=50)