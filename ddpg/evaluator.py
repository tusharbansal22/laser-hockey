import time
import numpy as np

def evaluate(agent, env, opponent, eval_episodes, quiet=False, action_mapping=None, evaluate_on_opposite_side=False):
    old_verbose = env.verbose
    env.verbose = not quiet

    rew_stats = []
    touch_stats = {}
    won_stats = {}
    lost_stats = {}

    for episode_counter in range(eval_episodes):
        total_reward = 0
        ob = env.reset()
        obs_agent2 = env.obs_agent_two()

        if (env.puck.position[0] < 5 and agent._config['mode'] == 'defense') or (
                env.puck.position[0] > 5 and agent._config['mode'] == 'shooting'
        ):
            continue

        touch_stats[episode_counter] = 0
        won_stats[episode_counter] = 0
        lost_stats[episode_counter] = 0

        for step in range(env.max_timesteps):
            if evaluate_on_opposite_side:
                a1, a2 = (opponent.act(ob), agent.act(obs_agent2, eps=0)) if agent._config['mode'] in ['defense', 'normal'] else ([0, 0, 0, 0], opponent.act(ob))
                a1 = a1 if isinstance(a1, np.ndarray) else action_mapping[a1]
                a2 = action_mapping[a2]

            else:
                a1, a2 = (agent.act(ob, eps=0), opponent.act(obs_agent2)) if agent._config['mode'] in ['defense', 'normal'] else (action_mapping[agent.act(ob, eps=0)], [0, 0, 0, 0])
                a1 = action_mapping[a1] if isinstance(a1, np.ndarray) else a1
                a2 = a2 if isinstance(a2, np.ndarray) else action_mapping[a2]

            (ob_new, reward, done, _info) = env.step(np.hstack([a1, a2]))
            ob, obs_agent2 = ob_new, env.obs_agent_two()

            if not evaluate_on_opposite_side:
                touch_stats[episode_counter] = 1 if _info['reward_touch_puck'] > 0 else 0
                total_reward += -reward if evaluate_on_opposite_side else reward

            if agent._config['show']:
                time.sleep(0.01)
                env.render()
            
            if done:
                won_stats[episode_counter] = 1 if env.winner == 1 else 0
                lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                break

        rew_stats.append(total_reward)
        if not quiet:
            agent.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon=0,
                                            touched=touch_stats[episode_counter])

    if not quiet:
        agent.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

    env.verbose = old_verbose
    return np.mean(rew_stats), np.mean(list(touch_stats.values())), np.mean(list(won_stats.values())), np.mean(list(lost_stats.values()))
