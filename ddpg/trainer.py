from collections import defaultdict
import numpy as np
import time

import sys
from utils import utils
from laserhockey import hockey_env as h_env

sys.path.insert(0, '.')
sys.path.insert(1, '..')
from base.evaluator import evaluate



class DDPGTrainer:
    def __init__(self, logger, config) -> None:
        self.logger = logger
        self._config = config


    def train(self, agent, opponents, env, eval):

        epsilon = 0.95
        epsilon_decay = 0.95
        min_epsilon = 0.06
        iter_fit = 30
        episode_counter = 1
        total_step_counter = 0

        rew_stats = []
        loss_stats = []
        lost_stats = {}
        touch_stats = {}
        won_stats = {}
        eval_stats = {
            'reward': [],
            'touch': [],
            'won': [],
            'lost': []
        }
        while episode_counter <= 10:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)
            total_reward = 0
            touched = 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0
            opponent = utils.poll_opponent(opponents)

            first_time_touch = 1
            for step in range(100):
                a1 = agent.act(ob, eps=epsilon)
                if self._config['mode'] == 'defense':
                    a2 = opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    a2 = opponent.act(obs_agent2)
                (ob_new, reward, done, _info) = env.step(np.hstack([a1, a2]))
                touched = max(touched, _info['reward_touch_puck'])
                current_reward = reward + 5 * _info['reward_closeness_to_puck'] - (
                        1 - touched) * 0.1 + touched * first_time_touch * 0.1 * step


                total_reward += current_reward

                first_time_touch = 1 - touched
                agent.store_transition((ob, a1, current_reward, ob_new, done))


                if self._config['show']:
                    time.sleep(0.01)
                    env.render()

                if touched > 0:
                    touch_stats[episode_counter] = 1

                if done:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                    break

                ob = ob_new
                obs_agent2 = env.obs_agent_two()
                total_step_counter += 1

            loss_stats.extend(agent.train(iter_fit=iter_fit, total_step_counter=episode_counter))

            rew_stats.append(total_reward)

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon)

            if episode_counter % 5 == 0:
                agent.eval()

                rew, touch, won, lost = evaluate(agent, env, h_env.BasicOpponent(weak=True),5, quiet=True)
                agent.train_mode()

                eval_stats['reward'].append(rew)
                eval_stats['touch'].append(touch)
                eval_stats['won'].append(won)
                eval_stats['lost'].append(lost)
                self.logger.save_model(agent, f'a-{episode_counter}.pk l')

                self.logger.plot_intermediate_stats(eval_stats, show=False)


            agent.schedulers_step()
            episode_counter += 1

        if self._config['show']:
            env.close()

        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

        self.logger.info('Saving training statistics...')

        self.logger.plot_running_mean(list(rew_stats), 'Total reward', 'total-reward.pdf', show=False)

        self.logger.plot_intermediate_stats(eval_stats, show=False)

        self.logger.plot_running_mean(list(loss_stats), 'Loss', 'loss.pdf', show=False)

        self.logger.save_model(agent, 'agent.pkl')

        print(eval_stats['won'])

        if eval:
            agent.eval()
            agent._config['show'] = True
            evaluate(agent, env, h_env.BasicOpponent(weak=False), 5)
            agent.train_mode()