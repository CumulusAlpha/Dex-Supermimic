"""Regression checks for upstream PPO parity, KL and playback compatibility."""

import copy
import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from dex_supermimic.rl.ppo import LoggedPPO, playback_actor_state
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor, Critic
from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner


def observations(n=8):
    return TensorDict({"policy": torch.randn(n, 7), "critic": torch.randn(n, 9)}, batch_size=[n])


def make_algorithm():
    obs = observations()
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        obs, groups, "actor", 3, hidden_dims=[16, 12], obs_normalization=True,
        distribution_cfg={
            "class_name": "rsl_rl.modules:HeteroscedasticGaussianDistribution",
            "init_std": 0.5, "std_range": (0.2, 50.0), "std_type": "scalar",
        },
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16, 12], obs_normalization=True)
    storage = RolloutStorage("rl", 8, 4, obs, [3])
    alg = PPO(
        actor, critic, storage, num_learning_epochs=2, num_mini_batches=2,
        schedule="fixed", learning_rate=1e-4, gamma=0.996, lam=0.95,
        entropy_coef=0.001, value_loss_coef=0.5, use_clipped_value_loss=False,
    )
    with torch.inference_mode():
        for t in range(4):
            alg.act(obs)
            obs = observations()
            # Include actual terminal and timeout transitions in the comparison.
            dones = torch.zeros(8, dtype=torch.long)
            timeouts = torch.zeros(8, dtype=torch.bool)
            if t == 2:
                dones[:2] = 1
                timeouts[1] = True
            alg.process_env_step(obs, torch.randn(8), dones, {"time_outs": timeouts})
        alg.compute_returns(obs)
    return alg


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_diagnostic_preserves_upstream_update_and_rng(self):
        torch.manual_seed(7)
        upstream = make_algorithm()
        logged = copy.deepcopy(upstream)
        logged.__class__ = LoggedPPO  # Same models, optimizer, GAE and rollout.
        torch.manual_seed(42)
        expected_losses = upstream.update()
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        actual_losses = logged.update()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key, value in expected_losses.items():
            self.assertEqual(value, actual_losses[key])
        for expected, actual in zip(
            list(upstream.actor.parameters()) + list(upstream.critic.parameters()),
            list(logged.actor.parameters()) + list(logged.critic.parameters()),
        ):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(upstream.optimizer.state.values(), logged.optimizer.state.values()):
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        with torch.inference_mode():
            obs = logged.storage.observations.flatten(0, 1)
            logged.actor.distribution.update(logged.actor.mlp(logged.actor.get_latent(obs)))
            old_mean, old_std = (p.flatten(0, 1) for p in logged.storage.distribution_params)
            expected_kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(old_mean, old_std), logged.actor.distribution._distribution,
            ).sum(-1).double().mean().item()
        self.assertGreater(actual_losses["kl_rollout_to_updated"], 0)
        self.assertAlmostEqual(actual_losses["kl_rollout_to_updated"], expected_kl, places=5)

    def test_native_checkpoint_playback_matches_upstream_mean(self):
        obs = observations()
        actor = MLPModel(
            obs, {"actor": ["policy"]}, "actor", 20,
            hidden_dims=[512, 256, 128], obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules:HeteroscedasticGaussianDistribution",
                "init_std": 0.5, "std_range": (0.2, 50.0), "std_type": "scalar",
            },
        )
        actor.update_normalization(obs)
        actor.eval()
        restored = GaussianPPOActor(7, 20)
        restored.load_state_dict(playback_actor_state(actor.state_dict()))
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored.mean(obs["policy"]), actor(obs), rtol=0, atol=0)

    def test_legacy_kl_uses_stored_rollout_distribution(self):
        # A shifted stored policy and lr=0 make the expected KL independent of
        # minibatch order, isolating the old=current regression in ppo_update().
        torch.manual_seed(4)
        actor = GaussianPPOActor(7, 3, hidden=(8, 8, 8))
        critic = Critic(9, empirical_norm=True)
        obs, critic_obs = torch.randn(4, 2, 7), torch.randn(4, 2, 9)
        with torch.no_grad():
            dist = actor.distribution(obs)
            old_mean, old_std = dist.mean + 0.4, dist.stddev * 1.3
            old = torch.distributions.Normal(old_mean, old_std)
            actions = old.sample()
            values = critic(critic_obs)
            expected_kl = torch.distributions.kl_divergence(old, dist).sum(-1).mean().item()
        memory = dict(
            obs=obs, critic_obs=critic_obs, actions=actions,
            action_mean=old_mean, action_std=old_std,
            log_prob=old.log_prob(actions).sum(-1), value=values,
            reward=torch.randn(4, 2), terminated=torch.zeros(4, 2), truncated=torch.zeros(4, 2),
        )
        runner = object.__new__(DexSuperMimicRunner)
        runner.actor, runner.ppo_critic, runner.ppo_memory = actor, critic, memory
        runner.cfg = SimpleNamespace(
            ppo_gamma=0.996, lam=0.95, ppo_normalize_advantage=True,
            ppo_minibatches=2, ppo_epochs=2, ppo_clip=0.2,
            ppo_value_coef=0.5, ppo_entropy=0.001, ppo_max_grad_norm=1,
        )
        runner.device = "cpu"
        runner._critic_observation = lambda: critic_obs[-1]
        runner.optimizers = {"ppo": torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0)}
        result = runner.ppo_update()
        self.assertGreater(result["kl"].item(), 0)
        self.assertAlmostEqual(result["kl"].item(), expected_kl, places=5)


if __name__ == "__main__":
    unittest.main()
