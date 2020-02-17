from textwrap import dedent

import streamlit as st
import numpy as np
import pandas as pd

"""
# Multi Armed Bandits
"""


def train_agents(agents, environments):
    states = environments.get_initial_states()
    while True:
        actions = agents.choose_actions(states)
        (rewards, states, metrics) = environments.evaluate(actions)
        agents.learn(rewards)
        yield metrics


class GaussianBandits:
    def __init__(
        self,
        bandit_count,
        arm_count=10,
        bandit_mean=0.0,
        bandit_scale=1.0,
        arm_scale=1.0,
        update_mean=None,
        update_scale=None,
    ):
        self.arm_means = np.random.normal(
            loc=bandit_mean, scale=bandit_scale, size=(arm_count, bandit_count)
        )
        self.arm_scale = arm_scale
        self.update_mean = update_mean
        self.update_scale = update_scale

    def get_initial_states(self):
        return None

    def evaluate(self, actions):
        flat_actions = actions.view()
        flat_actions.shape = (-1,)

        chosen_arm_means = flat_actions.choose(self.arm_means).reshape(actions.shape)
        rewards = chosen_arm_means + np.random.normal(
            loc=0.0, scale=self.arm_scale, size=chosen_arm_means.shape
        )
        new_state = None
        was_best_action_taken = (flat_actions == self.arm_means.argmax(axis=0)).reshape(
            actions.shape
        )
        metrics = (rewards, was_best_action_taken)

        # Update arm means
        if self.update_mean is not None:
            self.arm_means = self.arm_means + np.random.normal(
                loc=self.update_mean, scale=self.update_scale, size=self.arm_means.shape
            )

        return (rewards, new_state, metrics)


class EpsilonGreedyAgents:
    """
    ε-Greedy Agent.
    """

    def __init__(self, arm_count, epsilons, step_sizes, initial_estimates, trial_count):
        if not len(epsilons) == len(step_sizes):
            raise ValueError(f"epsilons and step_sizes must be the same length")
        if not len(epsilons) == len(initial_estimates):
            raise ValueError(f"epsilons and initial_estimates must be the same length")
        self.unique_agent_count = len(epsilons)
        self.trial_count = trial_count
        self.total_agent_count = self.unique_agent_count * self.trial_count

        self.epsilons = np.array(epsilons).reshape(-1, 1)
        self.step_sizes = np.repeat([(s or 1) for s in step_sizes], trial_count)
        self.use_sample_average = np.repeat(
            [(0 if s else 1) for s in step_sizes], trial_count
        )
        self.expected_rewards = np.zeros(
            (arm_count, len(epsilons), trial_count)
        ) + np.array(initial_estimates).reshape(1, -1, 1)
        self.times_tried = np.zeros((arm_count, len(epsilons), trial_count), dtype=int)

        self.env_range = np.arange(trial_count * len(epsilons))
        self.short_descriptions = self.get_short_descriptions(epsilons, step_sizes, initial_estimates)

    def choose_actions(self, states):
        best_actions = self.expected_rewards.argmax(axis=0)
        should_exploit = np.random.uniform(size=best_actions.shape) >= self.epsilons
        random_actions = np.random.randint(10, size=best_actions.shape)

        self.last_actions = best_actions * should_exploit + random_actions * (
            1 - should_exploit
        )
        return self.last_actions

    def learn(self, rewards):
        # TODO there's a lot of mucking about with reshaping arrays here which
        # obscures the actual algorithm. I suspect that if I knew how to numpy
        # property, this could be cleaned up. It would be nice to revisit this.
        last_actions = self.last_actions.view()
        last_actions.shape = (-1,)
        rewards = rewards.view()
        rewards.shape = (-1,)
        times_tried = self.times_tried.view()
        times_tried.shape = (-1, self.env_range.shape[0])
        expected_rewards = self.expected_rewards.view()
        expected_rewards.shape = (-1, self.env_range.shape[0])

        times_tried[last_actions, self.env_range] += 1
        step_sizes = self.step_sizes / (
            times_tried[last_actions, self.env_range] ** self.use_sample_average
        )
        expected_rewards[last_actions, self.env_range] += step_sizes * (
            rewards - expected_rewards[last_actions, self.env_range]
        )

    def get_short_descriptions(self, epsilons, step_sizes, initial_estimates):
        description_parts = []
        if any(e != self.epsilons[0] for e in epsilons):
            description_parts.append(
                (f"ε = {e}" if e else "Greedy" for e in epsilons)
            )
        if any(s != step_sizes[0] for s in step_sizes):
            description_parts.append(
                (f"α = {s}" if s else "Sample-average" for s in step_sizes)
            )
        if any(i != initial_estimates[0] for i in initial_estimates):
            description_parts.append(
                (f"Q₁ = {i}" for i in initial_estimates)
            )
        return [", ".join(d) for d in zip(*description_parts)]


@st.cache(
    hash_funcs={st.DeltaGenerator.DeltaGenerator: lambda x: None},
    suppress_st_warning=True,
)
def generate_metrics(
    agents,
    arm_count,
    bandit_mean,
    bandit_scale,
    arm_scale,
    update_mean,
    update_scale,
    step_count,
    visualise_progress,
    page_elements,
):
    environments = GaussianBandits(
        bandit_count=agents.total_agent_count,
        arm_count=arm_count,
        bandit_mean=bandit_mean,
        bandit_scale=bandit_scale,
        arm_scale=arm_scale,
        update_mean=update_mean,
        update_scale=update_scale,
    )

    raw_metric_data = [
        np.zeros((step_count + 1, agents.unique_agent_count)),
        np.zeros((step_count + 1, agents.unique_agent_count)),
    ]
    metrics = [
        pd.DataFrame(d, columns=agents.short_descriptions) for d in raw_metric_data
    ]

    for (step, metrics_for_step) in enumerate(train_agents(agents, environments)):
        if step >= step_count:
            break
        for (i, metric_type) in enumerate(metrics_for_step):
            raw_metric_data[i][step + 1] = metric_type.mean(axis=1)
        if step <= 20 or (step + 1) % (step_count // 20) == 0:
            visualise_progress([m[: step + 2] for m in metrics], page_elements)
    return metrics


def visualize_bandit_training(
    agents,
    arm_count,
    bandit_mean,
    bandit_scale,
    arm_scale,
    step_count,
    update_mean=None,
    update_scale=None,
):
    def visualise_progress(metrics, graph_blocks):
        for (graph_block, metric) in zip(graph_blocks, metrics):
            graph_block.line_chart(metric)

    graph_blocks = [st.empty(), st.empty()]
    metrics = generate_metrics(
        agents=agents,
        arm_count=arm_count,
        bandit_mean=bandit_mean,
        bandit_scale=bandit_scale,
        arm_scale=arm_scale,
        update_mean=update_mean,
        update_scale=update_scale,
        step_count=step_count,
        visualise_progress=visualise_progress,
        page_elements=graph_blocks,
    )
    visualise_progress(metrics, graph_blocks)


def figure_2_2():
    """
    Recreate Figure 2.2 from the book.
    """
    description = st.empty()
    trial_count = st.radio(
        "Number of trials", (20, 2000, 20000), format_func=lambda n: f"{n:,}", index=1
    )
    description.markdown(
        dedent(
            f"""
    ### Figure 2.2

    _Recreation of figure 2.2 from the book_

    Average performance of ε-greedy action-value methods on the 10-armed testbed. These data are
    averages over {trial_count:,} runs with different bandit problems. All methods used sample
    averages as their action-value estimates.
    """
        )
    )
    arm_count = 10
    agents = EpsilonGreedyAgents(
        arm_count=arm_count,
        epsilons=[0, 0.1, 0.01],
        step_sizes=[0, 0, 0],
        initial_estimates=[0, 0, 0],
        trial_count=trial_count,
    )
    visualize_bandit_training(
        agents=agents,
        arm_count=arm_count,
        bandit_mean=0.0,
        bandit_scale=1.0,
        arm_scale=1.0,
        step_count=1000,
    )


def exercise_2_5():
    """
    Solution to Exercise 2.5
    """
    st.markdown(
        dedent(
            """
    ### Exercise 2.5

    Design and conduct an experiment to demonstrate the difficulties that sample-average method
    have for nonstationary problems. Use a modified version of the 10-armed testbed in which all
    the q\*(a) start out equal and then take independent random walks (say by adding a normally
    distributed increment with mean zero and standard deviation 0.01 to all the q\*(a) on each
    step). Prepare plots like Figure 2.2 for an action-value method using sample averages,
    incrementally computed, and another action-value method using a constant step-size parameter,
    α = 0.1. Use ε = 0.1 and longer runs, say of 10,000 steps.
    """
        )
    )
    arm_count = 10
    agents = EpsilonGreedyAgents(
        arm_count=arm_count,
        epsilons=[0.1, 0.1],
        step_sizes=[0, 0.1],
        initial_estimates=[0, 0],
        trial_count=2000,
    )
    visualize_bandit_training(
        agents=agents,
        arm_count=arm_count,
        bandit_mean=0.0,
        bandit_scale=0.0,
        arm_scale=1.0,
        update_mean=0.0,
        update_scale=0.01,
        step_count=10000,
    )


def figure_2_3():
    """
    Recreate Figure 2.3 from the book.
    """
    st.markdown(
        dedent(
            f"""
    ### Figure 2.3

    _Recreation of figure 2.3 from the book_

    The effect of optimistic initial action-value estimates on the 10-armed testbed. Both methods
    used a constant step-size parameter, α = 0.1.
    """
        )
    )
    arm_count = 10
    agents = EpsilonGreedyAgents(
        arm_count=arm_count,
        epsilons=[0, 0.1],
        step_sizes=[0.1, 0.1],
        initial_estimates=[5, 0],
        trial_count=2000,
    )
    visualize_bandit_training(
        agents=agents,
        arm_count=arm_count,
        bandit_mean=0.0,
        bandit_scale=1.0,
        arm_scale=1.0,
        step_count=1000,
    )


if __name__ == "__main__":
    figure_2_2()
    exercise_2_5()
    figure_2_3()
