import numpy as np
import random
from agents.random_agent import RandomAgent


class GenerativeOpponentAgent(RandomAgent):
    """
    Generative opponent that learns an action distribution
    and samples from it (non-stationary behaviour)
    """

    def __init__(self, name, log_directory):
        super().__init__(name=name, log_directory=log_directory)
        self.action_counts = {}
        self.total_actions = 0

    def request_action(self, observation):
        possible_actions = observation["possible_actions"]

        # If no data yet, act randomly
        if self.total_actions < 20:
            action = random.randrange(len(possible_actions))
            return action

        # Build probability distribution
        probs = []
        for i in range(len(possible_actions)):
            probs.append(self.action_counts.get(i, 1))

        probs = np.array(probs, dtype=np.float32)
        probs /= probs.sum()

        action = np.random.choice(len(possible_actions), p=probs)
        return action

    def update_player_action(self, payload):
        if "action_index" in payload:
            idx = payload["action_index"]
            self.action_counts[idx] = self.action_counts.get(idx, 0) + 1
            self.total_actions += 1