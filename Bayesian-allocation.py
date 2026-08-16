"""This is a learning note/sample code when I prepare the following application problem for SPAR program.
Problem:
Allocate 100 computation units among 10 research directions. 

First we assume that all directions are independent, and each direction has a hidden true mean success probability drawn from a Beta prior. 
Each allocation unit is a Bernoulli trial with the hidden mean as its success probability. The goal is to maximize the total number of successes over 100 rounds.

We shall compare three allocation methods:
1. equal allocation;
2. Thompson sampling;
3. Monte Carlo rollout using Thompson sampling as its continuation policy.
by 3 rounds of simulation. In our experiment, ordinary Thomapson and rollout methods are on a par, but both are better than equal allocation. Moreover, in the experiment, the rollout
policy is not guaranteed to outperform Thompson sampling in a small run, probably because the number of rollouts is not large enough.

Reference: (1). T. Lattimore and C. Szepesvari,
    Bandit Algorithms,
    Cambridge University Press, 2020,
    Chapters 34–36.
    https://doi.org/10.1017/9781108571401
    (2). D. P. Bertsekas,
    A Course in Reinforcement Learning, 2nd edition,
    Athena Scientific, 2025,
    Chapters 1–2.
    https://www.mit.edu/~dimitrib/RLbook.html """


import argparse
from pathlib import Path

import numpy as np


POLICIES = ("equal", "thompson", "rollout")


class BetaPosterior:
    def __init__(self, number_of_arms, prior_alpha=1.0, prior_beta=1.0):
        self.alpha = np.full(number_of_arms, prior_alpha, dtype=float)
        self.beta = np.full(number_of_arms, prior_beta, dtype=float)
    
    @property 
    def number_of_arms(self):
        return len(self.alpha)

    def means(self):
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng, size=None):
        if size is None:
            return rng.beta(self.alpha, self.beta)
        return rng.beta(
            self.alpha,
            self.beta,
            size=(size, self.number_of_arms),
        )

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


class BanditEnvironment:
    
    def __init__(self, true_means, reward_table):
        self._true_means = true_means
        self._reward_table = reward_table
        self.pull_counts = np.zeros(len(true_means), dtype=int)

    def pull(self, arm):
        pull_number = self.pull_counts[arm]
        reward = self._reward_table[arm, pull_number]
        self.pull_counts[arm] += 1
        return int(reward)

    def pseudo_regret(self, arm):
        return float(np.max(self._true_means) - self._true_means[arm])

    def allocations_by_rank(self):
        best_to_worst = np.argsort(-self._true_means)
        return self.pull_counts[best_to_worst]


def random_argmax(values, rng):
    """Argmax with random tie-breaking."""
    candidates = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(candidates))


def thompson_action(posterior, rng):
    """Sample one possible mean per arm and choose the largest."""
    return int(np.argmax(posterior.sample(rng)))


def rollout_value(
    posterior,
    candidate_arm,
    remaining,
    simulated_means,
    reward_uniforms,
    rng,
):
    """Estimate reward from one candidate action followed by Thompson sampling."""
    number_of_rollouts = len(simulated_means)
    rows = np.arange(number_of_rollouts)

    # Each row is one simulated future; the real posterior is never changed.
    alpha = np.tile(posterior.alpha, (number_of_rollouts, 1))
    beta = np.tile(posterior.beta, (number_of_rollouts, 1))
    actions = np.full(number_of_rollouts, candidate_arm, dtype=int)
    returns = np.zeros(number_of_rollouts)

    for step in range(remaining):
        probabilities = simulated_means[rows, actions]
        rewards = (reward_uniforms[step] < probabilities).astype(int)
        returns += rewards

        alpha[rows, actions] += rewards
        beta[rows, actions] += 1 - rewards

        if step + 1 < remaining:
            posterior_draws = rng.beta(alpha, beta)
            actions = np.argmax(posterior_draws, axis=1)

    return returns.mean()


def rollout_action(posterior, remaining, number_of_rollouts, rng):
    """Choose the first action with the highest Monte Carlo rollout value."""
    if remaining == 1:
        return random_argmax(posterior.means(), rng)
    
    # Each latent mean vector stays fixed throughout its simulated future.
    
    simulated_means = posterior.sample(rng, number_of_rollouts)
    reward_uniforms = rng.random((remaining, number_of_rollouts))

    values = [
        rollout_value(
            posterior,
            arm,
            remaining,
            simulated_means,
            reward_uniforms,
            rng,
        )
        for arm in range(posterior.number_of_arms)
    ]
    return random_argmax(values, rng)


def make_reward_table(true_means, horizon, rng):
    """Generate each arm's potential rewards."""
    return rng.binomial(
        1,
        true_means[:, None],
        size=(len(true_means), horizon),
    )


def run_episode(
    policy,
    true_means,
    reward_table,
    prior_alpha,
    prior_beta,
    number_of_rollouts,
    rng,
):
    """Run one policy for one finite-horizon episode."""
    number_of_arms, horizon = reward_table.shape
    posterior = BetaPosterior(number_of_arms, prior_alpha, prior_beta)
    environment = BanditEnvironment(true_means, reward_table)
    equal_order = rng.permutation(number_of_arms)

    rewards = np.zeros(horizon, dtype=int)
    regrets = np.zeros(horizon)

    for t in range(horizon):
        remaining = horizon - t

        if policy == "equal":
            arm = int(equal_order[t % number_of_arms])
        elif policy == "thompson":
            arm = thompson_action(posterior, rng)
        elif policy == "rollout":
            arm = rollout_action(
                posterior,
                remaining,
                number_of_rollouts,
                rng,
            )
        else:
            raise ValueError(f"unknown policy: {policy}")

        reward = environment.pull(arm)
        posterior.update(arm, reward)
        rewards[t] = reward
        regrets[t] = environment.pseudo_regret(arm)

    return {
        "cumulative_reward": np.cumsum(rewards),
        "cumulative_pseudo_regret": np.cumsum(regrets),
        "allocations_by_rank": environment.allocations_by_rank(),
    }


def compare_policies(
    number_of_arms,
    horizon,
    episodes,
    number_of_rollouts,
    prior_alpha,
    prior_beta,
    seed,
):
    """Evaluate all policies on the same hidden worlds and reward streams."""
    results = {policy: [] for policy in POLICIES}
    environment_rng = np.random.default_rng(seed)
    policy_rngs = {
        policy: np.random.default_rng(seed + i + 1)
        for i, policy in enumerate(POLICIES)
    }

    for _ in range(episodes):
        # Draw each hidden mu once and keep it fixed throughout the episode.
        true_means = environment_rng.beta(
            prior_alpha,
            prior_beta,
            size=number_of_arms,
        )
        reward_table = make_reward_table(true_means, horizon, environment_rng)

        for policy in POLICIES:
            results[policy].append(
                run_episode(
                    policy,
                    true_means,
                    reward_table,
                    prior_alpha,
                    prior_beta,
                    number_of_rollouts,
                    policy_rngs[policy],
                )
            )

    return results


def print_summary(results):
    episodes = len(next(iter(results.values())))
    print(f"\nMean final performance over {episodes} episode(s)")
    print(f"{'policy':<12} {'reward':>10} {'pseudo-regret':>16}")
    print("-" * 40)

    for policy, runs in results.items():
        rewards = [run["cumulative_reward"][-1] for run in runs]
        regrets = [run["cumulative_pseudo_regret"][-1] for run in runs]
        print(f"{policy:<12} {np.mean(rewards):>10.2f} {np.mean(regrets):>16.2f}")

    print("\nMean allocation rank")
    for policy, runs in results.items():
        allocations = np.stack([run["allocations_by_rank"] for run in runs])
        print(f"{policy:<12} {np.round(allocations.mean(axis=0), 1)}")


def plot_results(results, output):
    import matplotlib.pyplot as plt

    colors = {"equal": "gray", "thompson": "royalblue", "rollout": "crimson"}
    labels = {
        "equal": "Equal allocation",
        "thompson": "Thompson sampling",
        "rollout": "TS + rollout",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for i, (policy, runs) in enumerate(results.items()):
        reward_curves = np.stack([run["cumulative_reward"] for run in runs])
        regret_curves = np.stack(
            [run["cumulative_pseudo_regret"] for run in runs]
        )
        allocations = np.stack([run["allocations_by_rank"] for run in runs])
        rounds = np.arange(1, reward_curves.shape[1] + 1)

        axes[0].plot(
            rounds,
            reward_curves.mean(axis=0),
            color=colors[policy],
            label=labels[policy],
        )
        axes[1].plot(
            rounds,
            regret_curves.mean(axis=0),
            color=colors[policy],
        )
        axes[2].bar(
            np.arange(1, allocations.shape[1] + 1) + 0.25 * (i - 1),
            allocations.mean(axis=0),
            width=0.24,
            color=colors[policy],
        )

    axes[0].set(
        title="Mean cumulative reward",
        xlabel="Round",
        ylabel="Successes",
    )
    axes[1].set(
        title="Mean cumulative pseudo-regret",
        xlabel="Round",
        ylabel="Pseudo-regret",
    )
    axes[2].set(
        title="Allocation by hidden rank",
        xlabel="Rank (1 = best)",
        ylabel="Allocations",
    )
    axes[2].set_xticks(np.arange(1, allocations.shape[1] + 1))
    figure.legend(loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(figure)





def main():
    number_of_arms = 10
    horizon = 100
    episodes = 3
    number_of_rollouts = 50
    prior_alpha = 1.0
    prior_beta = 1.0
    seed = 20260815

    results = compare_policies(
        number_of_arms,
        horizon,
        episodes,
        number_of_rollouts,
        prior_alpha,
        prior_beta,
        seed,
    )

    print_summary(results)
    plot_results(results, "bandit_comparison.png")


if __name__ == "__main__":
    main()