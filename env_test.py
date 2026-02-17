import sys
import os
import asyncio

# Add src directory to path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from rooms.room import Room
from agents.random_agent import RandomAgent


async def main():
    # Create a local room (no server)
    room = Room(
        run_remote_room=False,
        room_name="local_test_room",
        max_matches=1
    )

    # Create 4 random agents
    players = [
        RandomAgent(name=f"P{i}", log_directory=room.room_dir)
        for i in range(4)
    ]

    for p in players:
        room.connect_player(p)

    # Run one match
    await room.run()

    print("Final scores:", room.final_scores)


if __name__ == "__main__":
    asyncio.run(main())