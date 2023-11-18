import os
import torch
from laserhockey import hockey_env as h_env
from argparse import ArgumentParser
import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')

from ddpg.agent import DDPGAgent
from ddpg.trainer import DDPGTrainer
from utils.utils import *

parser = ArgumentParser()

parser.add_argument('--dry-run', help='Set if running only for sanity check', action='store_true')
parser.add_argument('--show', help='Set if want to render training process', action='store_true', default=True)
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true', default='true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='normal')

# Training params

parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=100)
parser.add_argument('--max_steps', help='Max steps for training', type=int, default=100)
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=100)

parser.add_argument('--evaluate_every',
                    help='# of episodes between evaluating agent during the training', type=int, default=20)
parser.add_argument('--learning_rate_actor', help='Learning rate for the actor network', type=float, default=0.0001)
parser.add_argument('--learning_rate_critic', help='Learning rate for the critic network', type=float, default=0.0001)
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+', default='20000')
parser.add_argument('--discount', help='Discount factor for future rewards', type=float, default=0.95)
parser.add_argument('--eps', help='Epsilon', type=float, default=0.95)
parser.add_argument('--epsilon_decay', help='Epsilon decay', type=float, default=0.95)
parser.add_argument('--min_epsilon', help='min_epsilon', type=float, default=0.06)
parser.add_argument('--iter_fit', help='Fit every n iterations', type=float, default=30)
parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
parser.add_argument('--tau', help='Tau value for soft updates', type=float, default=0.005)
parser.add_argument('--update_target_every', help='# of steps between updating target net', type=int, default=1)
parser.add_argument('--per', help='Utilize Prioritized Experience Replay', action='store_true', default=False)
parser.add_argument('--per_alpha', help='Alpha for PER', type=float, default=0.6)

parser.add_argument('--noise', help='# size of a noise', type=int, default=0.2)
parser.add_argument('--noise_clip', help='# noise limits', type=int, default=0.3)

parser.add_argument('--filename', help='Path to the pretrained model', default='logs/agents/a-10000.pkl')

opts = parser.parse_args()

if __name__ == '__main__':
    if opts.dry_run:
        opts.max_episodes = 10

    if opts.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif opts.mode == 'shooting':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif opts.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help')

    opts.device = torch.device('cpu')
    logger = Logger(prefix_path=os.path.dirname(os.path.realpath(__file__)) + '/logs',
                    mode=opts.mode,
                    cleanup=False,
                    quiet=opts.q)

    opponents = [h_env.BasicOpponent(weak=False)]
    env = h_env.HockeyEnv(mode=mode, verbose=(not opts.q))
    agent = DDPGAgent(
        logger=logger,
        obs_dim=env.observation_space.shape,
        action_space=env.action_space,
        userconfig=vars(opts)
    )

    trainer = DDPGTrainer(logger, vars(opts))
    trainer.train(agent, opponents, env, opts.evaluate)
