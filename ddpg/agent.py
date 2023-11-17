import sys
import numpy as np

import torch
import torch.nn.functional as F
from pathlib import Path
import pickle

sys.path.insert(0, '.')
sys.path.insert(1, '..')
# from base.agent import Agent
from ddpg.models import Actor, Critic, TwinCritic
# from base.experience_replay import UniformExperienceReplay
from utils.utils import soft_update

class ExperienceReplay:

    def __init__(self, max_size=100000):
        self._transitions = np.asarray([])
        self._current_idx = 0
        self.size = 0
        self.max_size = max_size

    @staticmethod
    def clone_buffer(new_buffer, maxsize):
        old_transitions = deepcopy(new_buffer._transitions)
        buffer = UniformExperienceReplay(max_size=maxsize)
        for t in old_transitions:
            buffer.add_transition(t)

        return buffer

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self._transitions = np.asarray(blank_buffer)

        self._transitions[self._current_idx, :] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self._current_idx = (self._current_idx + 1) % self.max_size

    def preload_transitions(self, path):
        for file in os.listdir(path):
            if file.endswith(".npz"):
                fpath = os.path.join(path, file)

                with np.load(fpath, allow_pickle=True) as d:
                    np_data = d['arr_0'].item()

                    if (
                        # Add a fancy condition
                        True
                    ):
                        transitions = utils.recompute_rewards(np_data, 'Dimitrije_Antic_-_SAC_ЈУГО')
                        for t in transitions:
                            tr = (
                                t[0],
                                t[1],
                                float(t[3]),
                                t[2],
                                bool(t[4]),
                            )
                            self.add_transition(tr)

        print(f'Preloaded data... Buffer size {self.size}.')

    def sample(self, batch_size):
        raise NotImplementedError("Implement the sample method")


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, max_size=100000):
        super(UniformExperienceReplay, self).__init__(max_size)

    def sample(self, batch_size):
        if batch_size > self.size:
            batch_size = self.size

        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return self._transitions[indices, :]


class DDPGAgent(object):

    def __init__(self, logger, obs_dim, action_space, userconfig):

        self.logger=logger,
        self.obs_dim=obs_dim,
        self.action_dim=action_space.shape[0],
        self.userconfig=userconfig
        self._config = {
            "eps": 0.05,
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate_actor": 0.0002,
            "learning_rate_critic": 0.0002,

            "hidden_sizes": [256, 256],
            'tau': 0.0001
        }
        self.buffer = UniformExperienceReplay(max_size=self._config['buffer_size'])
        

        self._observation_dim = obs_dim
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        

        self._config.update(userconfig)
        self._eps = self._config['eps']
        self._tau = self._config['tau']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eval_mode = False

        if self._config['lr_milestones'] is None:
            raise ValueError('lr_milestones argument cannot be None!\nExample: --lr_milestones=100 200 300')

        lr_milestones = [int(x) for x in (self._config['lr_milestones'][0]).split(' ')]

        # Critic
        self.critic = Critic(self._observation_dim, self._action_n,
                             hidden_sizes=self._config['hidden_sizes'],
                             learning_rate=self._config['learning_rate_critic'],
                             lr_milestones=lr_milestones,
                             lr_factor=self._config['lr_factor'],
                             device=self._config['device'])
        self.critic_target = Critic(self._observation_dim, self._action_n,
                                    hidden_sizes=self._config['hidden_sizes'],
                                    learning_rate=self._config['learning_rate_critic'],
                                    lr_milestones=lr_milestones,
                                    lr_factor=self._config['lr_factor'],
                                    device=self._config['device'])

        # Actor
        self.actor = Actor(self._observation_dim, self._action_n,
                           hidden_sizes=self._config['hidden_sizes'],
                           learning_rate=self._config['learning_rate_actor'],
                           lr_milestones=lr_milestones,
                           lr_factor=self._config['lr_factor'],
                           device=self._config['device'])
        self.actor_target = Actor(self._observation_dim, self._action_n,
                                  hidden_sizes=self._config['hidden_sizes'],
                                  learning_rate=self._config['learning_rate_actor'],
                                  lr_milestones=lr_milestones,
                                  lr_factor=self._config['lr_factor'],
                                  device=self._config['device'])

    def eval(self):
        self.eval_mode = True

    def train_mode(self):
        self.eval_mode = False

    def act(self, observation, eps=0, evaluation=False):
        state = torch.from_numpy(observation).float().to(self.device)
        if eps is None:
            eps = self._eps

        if np.random.random() > eps or evaluation:

            action = self.actor.forward(state)
            action = action.detach().cpu().numpy()[0]
        else:
            action = self._action_space.sample()[:4]

        return action

    def schedulers_step(self):
        self.critic.lr_scheduler.step()
        self.critic_target.lr_scheduler.step()
        self.actor.lr_scheduler.step()
        self.actor_target.lr_scheduler.step()

    def store_transition(self, transition):
        self.buffer.add_transition(transition)


    @staticmethod
    def load_model(fpath):
        with open(Path(fpath), 'rb') as inp:
            return pickle.load(inp)

    def train(self, total_step_counter, iter_fit=32):
        losses = []

        for i in range(iter_fit):
            data = self.buffer.sample(batch_size=self._config['batch_size'])
            s = torch.FloatTensor(
                np.stack(data[:, 0])
            ).to(self.device)

            s_next = torch.FloatTensor(
                np.stack(data[:, 3])
            ).to(self.device)
            a = torch.FloatTensor(
                np.stack(data[:, 1])[:, None]
            ).squeeze(dim=1).to(self.device)

            rew = torch.FloatTensor(
                np.stack(data[:, 2])[:, None]
            ).squeeze(dim=1).to(self.device)

            done = torch.FloatTensor(
                np.stack(data[:, 4])[:, None]
            ).squeeze(dim=1).to(self.device)  # done flag

            Q_target = self.critic(s, a).squeeze(dim=1).to(self.device)
            a_next = self.actor_target.forward(s_next)
            Q_next = self.critic_target.forward(s_next, a_next).squeeze(dim=1).to(self.device)
            # target
            targets = rew + self._config['discount'] * Q_next * (1.0 - done)

            # optimize critic
            targets = targets.to(self.device)

            critic_loss = self.critic.loss(Q_target.float(), targets.float())
            losses.append(critic_loss)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            actions = self.actor.forward(s)
            actor_loss = - self.critic.forward(s, actions).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # update

            if (total_step_counter) % self._config['update_target_every'] == 0:
                # optimize actor
                soft_update(self.critic_target, self.critic, self._tau)
                soft_update(self.actor_target, self.actor, self._tau)

        return losses
