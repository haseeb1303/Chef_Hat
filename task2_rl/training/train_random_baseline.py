import sys
import os
import asyncio
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(BASE_DIR, "src")
TASK2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(SRC_DIR)
sys.path.append(TASK2_DIR)
sys.path.append(os.path.join(TASK2_DIR, "agents"))
sys.path.append(os.path.join(TASK2_DIR, "output"))

from rooms.room import Room
from agents.random_agent import RandomAgent
from dqn_agent import DQNAgent
from log_results import save_results


class LearningAgent(RandomAgent):
    def __init__(self, name, log_directory, dqn_agent):
        super().__init__(name=name, log_directory=log_directory)
        self.dqn = dqn_agent
        self.last_state = None
        self.last_action = None

    def request_action(self, observation):
        state = np.concatenate([observation["hand"], observation["board"]])
        valid_actions = list(range(len(observation["possible_actions"])))
        action = self.dqn.select_action(state, valid_actions)
        self.last_state = state
        self.last_action = action
        return action

    def update_player_action(self, payload):
        if self.last_state is None:
            return

        cards_left = payload.get("cards_left", 0)
        reward = -0.5 * cards_left
        if cards_left == 0:
            reward += 10

        done = cards_left == 0

        next_state = np.concatenate([
            payload["observation_after"]["hand"],
            payload["observation_after"]["board"]
        ])

        self.dqn.store((self.last_state, self.last_action, reward, next_state, done))
        self.dqn.train_step()

        self.last_state = None
        self.last_action = None


async def main():
    dqn = DQNAgent(state_dim=28, action_dim=200)

    room = Room(
        run_remote_room=False,
        room_name="dqn_vs_random",
        max_matches=120
    )

    learner = LearningAgent("DQN", room.room_dir, dqn)

    opponents = [
        RandomAgent("R1", room.room_dir),
        RandomAgent("R2", room.room_dir),
        RandomAgent("R3", room.room_dir)
    ]

    room.connect_player(learner)
    for o in opponents:
        room.connect_player(o)

    await room.run()

    print("Final scores:", room.final_scores)
    save_results("dqn_vs_random", room.final_scores)


if __name__ == "__main__":
    asyncio.run(main())
