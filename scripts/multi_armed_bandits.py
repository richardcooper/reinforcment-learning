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
    ):
        self.arm_means = np.random.normal(
            loc=bandit_mean, scale=bandit_scale, size=(arm_count, bandit_count)
        )
        self.arm_scale = arm_scale

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

        return (rewards, new_state, metrics)


class EpsilonGreedyAgents:
    """
    ε-Greedy Agent.
    """

    def __init__(self, arm_count, epsilons, trial_count):
        self.unique_agent_count = len(epsilons)
        self.trial_count = trial_count
        self.total_agent_count = self.unique_agent_count * self.trial_count

        self.epsilons = np.array(epsilons).reshape(-1, 1)
        self.expected_rewards = np.zeros((arm_count, len(epsilons), trial_count))
        self.times_tried = np.zeros((arm_count, len(epsilons), trial_count), dtype=int)

        self.env_range = np.arange(trial_count * len(epsilons))

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

        expected_rewards[last_actions, self.env_range] += (
            rewards - expected_rewards[last_actions, self.env_range]
        ) / times_tried[last_actions, self.env_range]

    @property
    def short_descriptions(self):
        return [f"ε = {e}" for e in self.epsilons.flat]


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
    )

    raw_metric_data = [
        np.zeros((step_count+1, agents.unique_agent_count)),
        np.zeros((step_count+1, agents.unique_agent_count)),
    ]
    metrics = [
        pd.DataFrame(d, columns=agents.short_descriptions) for d in raw_metric_data
    ]

    for (step, metrics_for_step) in enumerate(train_agents(agents, environments)):
        if step >= step_count:
            break
        for (i, metric_type) in enumerate(metrics_for_step):
            raw_metric_data[i][step+1] = metric_type.mean(axis=1)
        if step <= 20 or (step + 1) % 50 == 0:
            visualise_progress([m[: step + 2] for m in metrics], page_elements)
    return metrics


def visualize_bandit_training(
    agents, arm_count, bandit_mean, bandit_scale, arm_scale, step_count
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
        step_count=step_count,
        visualise_progress=visualise_progress,
        page_elements=graph_blocks,
    )
    visualise_progress(metrics, graph_blocks)


def figure_2_2():
    """
    Recreate Figure 2.2 from the book.
    """
    st.markdown("### Figure 2.2")
    arm_count = 10
    agents = EpsilonGreedyAgents(
        arm_count=arm_count, epsilons=[0, 0.1, 0.01], trial_count=2000
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
