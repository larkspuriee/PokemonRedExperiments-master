import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

# Environment creation
def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    # Set session path and episode length
    # Creates a unique directory for saving logs, metrics, and visualizations for this session
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    # Maximum number of steps per episode
    ep_length = 5000

    print(f"Session path: {sess_path}")

    # Environment configuration
    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
    }

    num_cpu = 1
    env = make_env(0, env_config)()
    # Path to the pre-trained model checkpoint
    file_name = 'C:/Users/Acer/PycharmProjects/pythonProject/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip'

    # Checks if the model file exists
    if not exists(file_name):
        print(f"Checkpoint file {file_name} does not exist.")
        print("Creating new model and starting training...")
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        print(f"Loading model from: {file_name}")
        model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    episode_rewards = []
    exploration_steps = []
    total_reward = 0
    explored_areas = set()

    obs, info = env.reset()
    # Runs for 10 episodes.
    for episode in range(10):  # Limit to 10 episodes
        print(f"Episode {episode + 1} started.")  # Track and display the episode number
        step_rewards = []
        total_steps = 0

        while total_steps < ep_length:
            action = 7
            # Tries to read agent_enabled.txt to determine if the AI should make decisions
            try:
                with open("agent_enabled.txt", "r") as f:
                    agent_enabled = f.readlines()[0].startswith("yes")
            except FileNotFoundError:
                agent_enabled = False
            if agent_enabled:
                action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(action)

            # Collect data for visualizations
            step_rewards.append(rewards)
            explored_areas.add(info.get('position', None))  # Example: track unique positions
            total_reward += rewards
            total_steps += 1

            env.render()
            # If the episode ends (via termination or truncation), it resets the environment and logs the episode’s metrics
            if terminated or truncated:
                break

        episode_rewards.append(sum(step_rewards))
        exploration_steps.append(len(explored_areas))  # Add unique positions explored
        obs, info = env.reset()

        print(f"Episode {episode + 1} completed.")  # Track and display the episode completion

    env.close()

    # Plot Total Rewards per Episode
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards Over Episodes")
    plt.savefig(sess_path / "rewards_over_episodes.png")
    plt.show()

    # Plot Average Reward Progression Within Episodes
    average_rewards = [np.mean(episode_rewards[:i + 1]) for i in range(len(episode_rewards))]
    plt.figure()
    plt.plot(average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Progression")
    plt.savefig(sess_path / "average_rewards_progression.png")
    plt.show()

    # Stacked Bar Chart for Reward Contributions (Example with Random Data)
    rewards_contributions = np.random.rand(len(episode_rewards), 3)
    labels = [f"Episode {i + 1}" for i in range(len(episode_rewards))]

    plt.figure()
    plt.bar(labels, rewards_contributions[:, 0], label="Exploration")
    plt.bar(labels, rewards_contributions[:, 1], bottom=rewards_contributions[:, 0], label="Task Completion")
    plt.bar(labels, rewards_contributions[:, 2], bottom=rewards_contributions[:, 0] + rewards_contributions[:, 1],
            label="Other")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Reward Contribution Per Episode")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(sess_path / "reward_contributions.png")
    plt.show()

    print("All visualizations saved to:", sess_path)
