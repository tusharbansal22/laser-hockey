from argparse import ArgumentParser
from laserhockey import hockey_env as h_env
import os
import sys
from utils.utils import *
from evaluator import evaluate

sys.path.extend(['.', '..'])

parser = ArgumentParser()
parser.add_argument('--filename', default=None)
parser.add_argument('--mode', default='normal', choices=['normal', 'shooting', 'defense'])
parser.add_argument('--show', action='store_true', default=False, help='Render training process')
parser.add_argument('--q', action='store_true', help='Quiet mode (no prints)')
parser.add_argument('--opposite', action='store_true', default=False, help='Evaluate agent on opposite side')
opts = parser.parse_args()

modes = {
    'normal': h_env.HockeyEnv_BasicOpponent.NORMAL,
    'shooting': h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING,
    'defense': h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
}

if __name__ == '__main__':
    mode = modes.get(opts.mode, None)
    if mode is None:
        raise ValueError('Unknown training mode. See --help')

    logger = Logger(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs'), mode=opts.mode, quiet=opts.q)
    q_agent = logger.load_model(opts.filename)
    q_agent._config['show'] = opts.show
    env = h_env.HockeyEnv(mode=mode)
    q_agent.eval()
    opponent = h_env.BasicOpponent(weak=True)
    evaluate(q_agent, env, opponent, 100, evaluate_on_opposite_side=opts.opposite)
