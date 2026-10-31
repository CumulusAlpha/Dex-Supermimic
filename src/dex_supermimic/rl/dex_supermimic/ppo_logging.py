"""PPO episode statistics and console output, following Wuji's rsl-rl runner."""

from collections import defaultdict, deque
from datetime import timedelta

import torch


class PPOLogStats:
    def __init__(self, num_envs, device):
        self.reward_sum = torch.zeros(num_envs, device=device)
        self.episode_length = torch.zeros(num_envs, device=device)
        self.rewards = deque(maxlen=100)
        self.lengths = deque(maxlen=100)
        self.extras = defaultdict(list)

    @torch.no_grad()
    def process_step(self, rewards, dones, extras):
        # Track real episode returns before PPO's timeout bootstrap correction.
        self.reward_sum += rewards
        self.episode_length += 1
        ids = dones.nonzero(as_tuple=True)[0]
        self.rewards.extend(self.reward_sum[ids].cpu().tolist())
        self.lengths.extend(self.episode_length[ids].cpu().tolist())
        self.reward_sum[ids] = 0
        self.episode_length[ids] = 0
        for name, value in extras.get('episode', extras.get('log', {})).items():
            value = torch.as_tensor(value, device=self.reward_sum.device).detach().float()
            if value.numel():
                key = name if '/' in name else f'Episode/{name}'
                self.extras[key].append(value.reshape(-1).clone())

    def scalars(self):
        values = {key: torch.cat(samples).mean().item() for key, samples in self.extras.items()}
        self.extras.clear()
        if self.rewards:
            values['Train/mean_reward'] = sum(self.rewards) / len(self.rewards)
            values['Train/mean_episode_length'] = sum(self.lengths) / len(self.lengths)
        return values


def organize_ppo_scalars(stats, iteration):
    names = {
        'actor_loss': 'Loss/surrogate', 'critic_loss': 'Loss/value_function',
        'entropy': 'Loss/entropy', 'clip_fraction': 'Policy/clip_fraction',
        'kl': 'Policy/kl', 'explained_variance': 'Policy/explained_variance',
        'action_std': 'Policy/mean_std', 'lr': 'Loss/learning_rate',
        'reward': 'Train/mean_step_reward', 'drop_rate': 'Train/drop_rate',
        'horizon': 'Train/rollout_steps', 'sim_steps': 'Perf/total_steps',
        'collection_time': 'Perf/collection_time', 'learning_time': 'Perf/learning_time',
        'fps': 'Perf/total_fps', 'wall_seconds': 'Perf/time_elapsed',
        'eta': 'Perf/eta',
    }
    scalars = {names.get(key, key): float(value) for key, value in stats.items()}
    scalars['Train/iteration'] = iteration
    return scalars


def format_ppo_log(iteration, max_iterations, stats):
    lines = ['#' * 80, f' Learning iteration {iteration}/{max_iterations} '.center(80), '']
    fields = (
        ('Total steps', 'Perf/total_steps', ',.0f'),
        ('Steps per second', 'Perf/total_fps', ',.0f'),
        ('Collection time (s)', 'Perf/collection_time', '.3f'),
        ('Learning time (s)', 'Perf/learning_time', '.3f'),
        ('Learning rate', 'Loss/learning_rate', '.6g'),
        ('Mean surrogate loss', 'Loss/surrogate', '.4f'),
        ('Mean value function loss', 'Loss/value_function', '.4f'),
        ('Mean entropy', 'Loss/entropy', '.4f'),
        ('Clip fraction', 'Policy/clip_fraction', '.4f'),
        ('KL divergence', 'Policy/kl', '.6g'),
        ('Explained variance', 'Policy/explained_variance', '.4f'),
        ('Mean reward', 'Train/mean_reward', '.2f'),
        ('Mean episode length', 'Train/mean_episode_length', '.2f'),
        ('Mean step reward', 'Train/mean_step_reward', '.4f'),
        ('Mean action std', 'Policy/mean_std', '.4f'),
        ('Drop rate', 'Train/drop_rate', '.4f'),
    )
    for label, key, spec in fields:
        if key in stats:
            lines.append(f'{label + ":":>48} {stats[key]:{spec}}')
    for key in sorted(stats):
        if key.startswith(('Episode', 'Metrics/', 'Curriculum/', 'curriculum/')):
            lines.append(f'{key + ":":>48} {stats[key]:.4f}')
    iteration_time = stats['Perf/collection_time'] + stats['Perf/learning_time']
    lines.extend([
        '-' * 80,
        f'{"Iteration time:":>48} {iteration_time:.2f}s',
        f'{"Time elapsed:":>48} {timedelta(seconds=int(stats["Perf/time_elapsed"]))}',
        f'{"ETA:":>48} {timedelta(seconds=int(stats["Perf/eta"]))}',
    ])
    return '\n'.join(lines)
