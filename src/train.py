"""Methods for training an agent."""
import os
import datetime
import pandas as pd
from matplotlib import pyplot as plt
from gym.wrappers import Monitor
import gym_tetris
import gym_super_mario_bros


def train(
    env_id: str,
    output_dir: str,
    is_monitor: bool=False
) -> None:
    """
    Train an agent to actuate a certain environment.

    Args:
        env_id: the ID of the environment to play
        output_dir: the base directory to store results into
        is_monitor: whether to wrap the environment with a monitor

    Returns:
        None

    """
    # setup the output directory based on the environment ID and current time
    now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
    output_dir = '{}/{}/DeepQAgent/{}'.format(output_dir, env_id, now)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('writing results to {}'.format(repr(output_dir)))
    weights_file = '{}/weights.h5'.format(output_dir)

    # these are long to import and train is only ever called once during
    # an execution lifecycle. import here to save early execution time
    from src.environment.atari import build_atari_environment
    from src.agents import DeepQAgent
    from src.util import BaseCallback

    # TODO: replace this logic using an internal `wrap` command
    if 'Tetris' in env_id:
        env = gym_tetris.make(env_id)
        env = gym_tetris.wrap(env, clip_rewards=False)
    elif 'SuperMarioBros' in env_id:
        env = gym_super_mario_bros.make(env_id)
        env = gym_super_mario_bros.wrap(env, clip_rewards=False)
    else:
        env = build_atari_environment(env_id)

    # wrap the environment with a monitor if enabled
    if is_monitor:
        env = Monitor(env, '{}/monitor_train'.format(output_dir), force=True)

    # build the agent
    agent = DeepQAgent(env, replay_memory_size=int(7.5e5))
    # write some info about the agent's hyperparameters to disk
    with open('{}/agent.py'.format(output_dir), 'w') as agent_file:
        agent_file.write(repr(agent))

    # observe frames to fill the replay memory
    agent.observe()

    # train the agent
    try:
        callback = BaseCallback(weights_file)
        agent.train(frames_to_play=int(2.5e6), callback=callback)
    except KeyboardInterrupt:
        print('canceled training')

    # save the weights to disk
    agent.model.save_weights(weights_file, overwrite=True)

    # save the training results
    rewards = pd.Series(callback.scores)
    losses = pd.Series(callback.losses)
    rewards_losses = pd.concat([rewards, losses], axis=1)
    rewards_losses.columns = ['Reward', 'Loss']
    rewards_losses.index.name = 'Episode'
    rewards_losses.to_csv('{}/rewards_losses.csv'.format(output_dir))
    rewards_losses.plot(figsize=(12, 5), subplots=True)
    plt.savefig('{}/rewards_losses.pdf'.format(output_dir))

    # close the environment to perform necessary cleanup
    env.close()


# explicitly define the outward facing API of this module
__all__ = [train.__name__]