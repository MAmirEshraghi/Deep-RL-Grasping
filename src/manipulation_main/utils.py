import logging
import time
import numpy as np

# Import the RobotEnv to use its status for determining episode success.
from manipulation_main.gripperEnv.robot import RobotEnv

def run_agent(task, agent, stochastic=False, n_episodes=100, debug=False):
    """
    Run the trained agent on the given environment (task) for a specified number of episodes.
    Collects performance statistics such as reward, steps, success rate, and execution time.
    
    Args:
        task (gym.Env): The environment in which to run the agent.
        agent: The trained RL agent.
        stochastic (bool): If True, the agent will take stochastic actions.
        n_episodes (int): Number of episodes to run.
        debug (bool): If True, use the debug version of the episode run.
        
    Returns:
        Tuple of arrays: (rewards, steps, success_rates, timings)
    """
    # Initialize arrays to store statistics for each episode.
    rewards = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)
    success_rates = np.zeros(n_episodes)
    timings = np.zeros(n_episodes)
    
    # For vectorized environments, only one initial reset is required.
    obs = task.reset()

    # Run through the specified number of episodes.
    for i in range(n_episodes):
        # Record start time (CPU process time).
        start = time.process_time()
        
        # Run one episode: choose debug mode if requested.
        if not debug:
            s, r, sr = _run_episode(obs, task, agent, stochastic)
        else:
            s, r, sr = _run_episode_debug(task, agent, stochastic)
        
        # Record end time.
        end = time.process_time()

        # Store episode statistics.
        rewards[i] = np.sum(r)  # Sum of rewards over the episode.
        steps[i] = s            # Total steps taken in the episode.
        success_rates[i] = sr   # Whether the episode was successful.
        timings[i] = end - start  # Time taken for the episode.

        # Log episode results.
        logging.info('Episode %d/%d completed in %ds, %d steps and return %f\n and success rate %d',
                     i+1, n_episodes, timings[i], steps[i], rewards[i], success_rates[i])

    # Calculate mean statistics over all episodes.
    mean_reward = np.mean(rewards)
    mean_steps = np.mean(steps)
    mean_success_rate = np.mean(success_rates)
    mean_time = np.mean(timings)

    # Print the aggregated statistics.
    print('{:<13}{:>5.2f}'.format('Mean reward:', mean_reward))
    print('{:<13}{:>5.2f}'.format('Mean steps:', mean_steps))
    print('{:<13}{:>5.2f}'.format('Mean success rate:', mean_success_rate))
    print('{:<13}{:>5.2f}'.format('Mean time:', mean_time))

    # Return the collected statistics.
    return rewards, steps, success_rates, timings

def _run_episode_debug(task, agent, stochastic):
    """
    Run one episode in debug mode.
    Uses agent.act instead of agent.predict and provides extra logging for debugging.
    
    Args:
        task (gym.Env): The environment.
        agent: The RL agent.
        stochastic (bool): Whether to use stochastic actions.
        
    Returns:
        A tuple: (episode_steps, episode_rewards, success_flag)
    """
    # Reset the environment at the start of the episode.
    obs = task.reset()
    done = False

    while not done:
        # Optionally log the observation for debugging.
        # logging.debug('Observation: %s', obs)

        # Obtain an action from the agent using its act method.
        action = agent.act(obs, stochastic=stochastic)
        
        # Execute the action in the environment.
        obs, reward, done, _ = task.step(action)
        
        # Retrieve the robot's current pose (e.g., for additional debug info).
        position, _ = task.get_pose()
        robot_height = position[2]  # Extract the height (unused in this function).
        
        # Optionally log action and reward.
        # logging.debug('Action: %s', action)
        # logging.debug('Reward: %s\n', reward)

    # Return episode statistics: steps taken, total reward, and success flag.
    return task.episode_step, task.episode_rewards, task.status == task.Status.SUCCESS

def _run_episode(obs, task, agent, stochastic):
    """
    Run one episode using the standard mode.
    Uses agent.predict to get actions and retrieves episode statistics from task buffers.
    
    Args:
        obs (np.ndarray): The initial observation.
        task (gym.Env): The environment.
        agent: The RL agent.
        stochastic (bool): Whether to use stochastic actions.
        
    Returns:
        A tuple: (episode_steps, episode_rewards, success_flag)
    """
    done = False
    # Set deterministic to the opposite of stochastic.
    deterministic = not stochastic
    while not done:
        # Optionally log the observation.
        # logging.debug('Observation: %s', obs)

        # Predict the next action using the agent.
        action = agent.predict(obs, deterministic=deterministic)
        # Execute the action in the environment.
        # Note: action[0] is used because agent.predict returns a tuple (action, state).
        obs, reward, done, _ = task.step(action[0])
        # Optionally log action and reward.
        # logging.debug('Action: %s', action)
        # logging.debug('Reward: %s\n', reward)
    
    # Retrieve episode statistics from the task's buffer.
    # These are stored in task.buf_infos[0] which is a dict containing info about the episode.
    episode_steps = task.buf_infos[0]["episode_step"]
    episode_rewards = task.buf_infos[0]["episode_rewards"]
    # Check if the episode ended in success by comparing status to RobotEnv.Status.SUCCESS.
    success_flag = task.buf_infos[0]["status"] == RobotEnv.Status.SUCCESS
    
    return episode_steps, episode_rewards, success_flag
