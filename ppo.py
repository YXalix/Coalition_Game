# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from IRSs_world import IRSsWorldEnv
from torch.utils.tensorboard import SummaryWriter


time_slot = 0
U_LUs = 1000
U_LUs_Count = 0

U_ATK = 1000
U_ATK_Count = 0

U_IRSs_0 = 1000
U_IRSs_0_Count = 0

U_IRSs_1 = 1000
U_IRSs_1_Count = 0

U_LUs_IRSs = 2000
U_LUs_IRSs_Count = 0

U_ATK_IRSs = 2000
U_ATK_IRSs_Count = 0

C_LUs = 0
C_ATK = 0
C_IRSs_0 = 0
C_IRSs_1 = 0

# 0: LUs, ATK, IRSs
# 1: {LUs, IRSs}, ATK
# 2: LUs, {ATK, IRSs}
C_state = 0
C_state_old = 0


total_reward_LUs = 0
total_reward_ATK = 0
total_reward_IRSs = 0
time_slot = 0

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = IRSsWorldEnv()#gym.make(env_id, render_mode="rgb_array")
        else:
            env = IRSsWorldEnv()#gym.make(env_id)
        #env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        #env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env,party_id):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space(party_id).shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space(party_id).shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space(party_id).shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space(party_id).shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id, 0, args.capture_video, run_name, args.gamma)()

    agent_LUs = Agent(env,'LUs').to(device)
    agent_ATK = Agent(env,'ATK').to(device)
    agent_IRSs = Agent(env,'IRSs').to(device)

    optimizer_LUs = optim.Adam(agent_LUs.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer_ATK = optim.Adam(agent_ATK.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer_IRSs = optim.Adam(agent_IRSs.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_LUs = torch.zeros((args.num_steps, args.num_envs) + env.observation_space('LUs').shape).to(device)
    actions_LUs = torch.zeros((args.num_steps, args.num_envs) + env.action_space('LUs').shape).to(device)
    logprobs_LUs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_LUs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_LUs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_LUs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    obs_ATK = torch.zeros((args.num_steps, args.num_envs) + env.observation_space('ATK').shape).to(device)
    actions_ATK = torch.zeros((args.num_steps, args.num_envs) + env.action_space('ATK').shape).to(device)
    logprobs_ATK = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_ATK = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_ATK = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_ATK = torch.zeros((args.num_steps, args.num_envs)).to(device)

    obs_IRSs = torch.zeros((args.num_steps, args.num_envs) + env.observation_space('IRSs').shape).to(device)
    actions_IRSs = torch.zeros((args.num_steps, args.num_envs) + env.action_space('IRSs').shape).to(device)
    logprobs_IRSs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_IRSs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_IRSs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_IRSs = torch.zeros((args.num_steps, args.num_envs)).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs_LUs = next_obs['LUs'].reshape(args.num_envs, -1)
    next_obs_ATK = next_obs['ATK'].reshape(args.num_envs, -1)
    next_obs_IRSs = next_obs['IRSs'].reshape(args.num_envs, -1)
    next_obs_LUs = torch.Tensor(next_obs_LUs).to(device)
    next_obs_ATK = torch.Tensor(next_obs_ATK).to(device)
    next_obs_IRSs = torch.Tensor(next_obs_IRSs).to(device)

    next_done_LUs = torch.zeros(args.num_envs).to(device)
    next_done_ATK = torch.zeros(args.num_envs).to(device)
    next_done_IRSs = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer_LUs.param_groups[0]["lr"] = lrnow
            optimizer_ATK.param_groups[0]["lr"] = lrnow
            optimizer_IRSs.param_groups[0]["lr"] = lrnow

        C_state_old = C_state
        U_LUs_IRSs = 2000
        U_ATK_IRSs = 2000
        for step in range(0, args.num_steps):
            # stage 1 coalition formation
            # for LUs
            rate = 0.1 if C_state_old == 1 else 0
            if (U_LUs_IRSs + U_LUs - U_IRSs_0) / 2 >  U_LUs - rate * abs(U_LUs):
                C_LUs = 1
            else:
                C_LUs = 0

            # for ATK
            rate = 0.1 if C_state_old == 2 else 0
            if (U_ATK_IRSs + U_ATK - U_IRSs_1)/2 > U_ATK - rate * abs(U_ATK):
                C_ATK = 1
            else:
                C_ATK = 0

            # for IRSs
            if C_state_old == 1:
                if (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2 > (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2 - 0.2 * abs(U_ATK_IRSs + U_IRSs_1 - U_ATK):
                    if (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2 > U_IRSs_0 - 0.2 * abs(U_IRSs_0):
                        C_IRSs_0 = 1
                        C_IRSs_1 = 0
                    else:
                        C_IRSs_0 = 0
                        C_IRSs_1 = 0
                else:
                    if (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2 - 0.2 * abs(U_ATK_IRSs + U_IRSs_1 - U_ATK) > U_IRSs_0 - 0.2 * abs(U_IRSs_0):
                        C_IRSs_1 = 1
                        C_IRSs_0 = 0
                    else:
                        C_IRSs_1 = 0
                        C_IRSs_0 = 0
            elif C_state_old == 2:
                if (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2 > (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2 - 0.2 * abs(U_LUs_IRSs + U_IRSs_0 - U_LUs):
                    if (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2 > U_IRSs_1 - 0.2 * abs(U_IRSs_1):
                        C_IRSs_1 = 1
                        C_IRSs_0 = 0
                    else:
                        C_IRSs_1 = 0
                        C_IRSs_0 = 0
                else:
                    if (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2 - 0.2 * abs(U_LUs_IRSs + U_IRSs_0 - U_LUs) > U_IRSs_1 - 0.2 * abs(U_IRSs_1):
                        C_IRSs_0 = 1
                        C_IRSs_1 = 0
                    else:
                        C_IRSs_0 = 0
                        C_IRSs_1 = 0
            else:
                if (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2 > 0:
                    if (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2 > (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2:
                        C_IRSs_0 = 1
                        C_IRSs_1 = 0
                if (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2 > 0:
                    if (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2 > (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2:
                        C_IRSs_0 = 0
                        C_IRSs_1 = 1
                if (U_LUs_IRSs + U_IRSs_0 - U_LUs)/2 <= 0 and (U_ATK_IRSs + U_IRSs_1 - U_ATK)/2 <= 0:
                    C_IRSs_0 = 0
                    C_IRSs_1 = 0


            # gen coalition state
            if C_LUs and C_IRSs_0:
                C_state = 1
            elif C_ATK and C_IRSs_1:
                C_state = 2
            else:
                C_state = 0

            # set stage 1 information to env
            env.set_utilities([U_LUs,U_ATK,U_IRSs_0,U_IRSs_1,U_LUs_IRSs,U_ATK_IRSs])
            env.set_coalition(C_state_old, C_state)
            env.set_coalition_decisions([C_LUs,C_ATK,C_IRSs_0,C_IRSs_1])

            global_step += 1 * args.num_envs
            obs_LUs[step] = next_obs_LUs
            obs_ATK[step] = next_obs_ATK
            obs_IRSs[step] = next_obs_IRSs

            dones_LUs[step] = next_done_LUs
            dones_ATK[step] = next_done_ATK
            dones_IRSs[step] = next_done_IRSs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_LUs, logprob_LUs, _, value_LUs = agent_LUs.get_action_and_value(next_obs_LUs)
                values_LUs[step] = value_LUs.flatten()
            actions_LUs[step] = action_LUs
            logprobs_LUs[step] = logprob_LUs

            with torch.no_grad():
                action_ATK, logprob_ATK, _, value_ATK = agent_ATK.get_action_and_value(next_obs_ATK)
                values_ATK[step] = value_ATK.flatten()
            actions_ATK[step] = action_ATK
            logprobs_ATK[step] = logprob_ATK

            with torch.no_grad():
                action_IRSs, logprob_IRSs, _, value_IRSs = agent_IRSs.get_action_and_value(next_obs_IRSs)
                values_IRSs[step] = value_IRSs.flatten()
            actions_IRSs[step] = action_IRSs
            logprobs_IRSs[step] = logprob_IRSs

            # TRY NOT TO MODIFY: execute the game and log data.
            actions = {'LUs': actions_LUs[step].cpu().numpy(),
                          'ATK': actions_ATK[step].cpu().numpy(),
                          'IRSs': actions_IRSs[step].cpu().numpy()}
            next_obs, reward, terminated, truncated, infos = env.step(actions)
            done = np.logical_or(terminated, truncated).reshape((1, -1))
            rewards_LUs[step] = torch.tensor(reward['LUs']).to(device).view(-1)
            rewards_ATK[step] = torch.tensor(reward['ATK']).to(device).view(-1)
            rewards_IRSs[step] = torch.tensor(reward['IRSs']).to(device).view(-1)
            next_obs_LUs = torch.Tensor(next_obs['LUs'].reshape(1,-1)).to(device)
            next_done_LUs = torch.Tensor(done).to(device)
            next_obs_ATK, next_done_ATK = torch.Tensor(next_obs['ATK'].reshape(1,-1)).to(device), torch.Tensor(done).to(device)
            next_obs_IRSs, next_done_IRSs = torch.Tensor(next_obs['IRSs'].reshape(1,-1)).to(device), torch.Tensor(done).to(device)

            # update coalition utilities information
            U_LUs_IRSs = infos['utilities'][4]
            U_ATK_IRSs = infos['utilities'][5]

            if done:
                total_reward_LUs += rewards_LUs[step].item()
                total_reward_ATK += rewards_ATK[step].item()
                total_reward_IRSs += rewards_IRSs[step].item()
                time_slot += 1
                # update all utilities information
                if C_state == 0:
                    U_LUs = U_LUs*U_LUs_Count/(U_LUs_Count+1) + infos['LUs']/(U_LUs_Count+1)
                    U_ATK = U_ATK*U_ATK_Count/(U_ATK_Count+1) + infos['ATK']/(U_ATK_Count+1)
                    U_IRSs_0 = U_IRSs_0*U_IRSs_0_Count/(U_IRSs_0_Count+1) + infos['IRSs']/(U_IRSs_0_Count+1)
                    U_IRSs_1 = U_IRSs_1*U_IRSs_1_Count/(U_IRSs_1_Count+1) + infos['IRSs']/(U_IRSs_1_Count+1)
                    U_LUs_Count += 1
                    U_ATK_Count += 1
                    U_IRSs_0_Count += 1
                    U_IRSs_1_Count += 1
                    U_LUs_Count = min(U_LUs_Count,99)
                    U_ATK_Count = min(U_ATK_Count,99)
                    U_IRSs_0_Count = min(U_IRSs_0_Count,99)
                    U_IRSs_1_Count = min(U_IRSs_1_Count,99)
                elif C_state == 1:
                    U_ATK = U_ATK*U_ATK_Count/(U_ATK_Count+1) + infos['ATK']/(U_ATK_Count+1)
                    U_IRSs_1 = U_IRSs_1*U_IRSs_1_Count/(U_IRSs_1_Count+1) + infos['IRSs']/(U_IRSs_1_Count+1)
                    U_LUs_Count += 1
                    U_ATK_Count += 1
                    U_IRSs_1_Count += 1
                    U_LUs_IRSs_Count = min(U_LUs_IRSs_Count,5)
                    U_ATK_Count = min(U_ATK_Count,99)
                    U_IRSs_1_Count = min(U_IRSs_1_Count,99)
                elif C_state == 2:
                    U_LUs = U_LUs * U_LUs_Count / (U_LUs_Count + 1) + infos['LUs']/((U_LUs_Count + 1))
                    U_IRSs_0 = U_IRSs_0*U_IRSs_0_Count/(U_IRSs_0_Count+1) + infos['IRSs']/(U_IRSs_0_Count+1)
                    U_ATK_IRSs_Count = min(U_ATK_IRSs_Count,5)
                    U_LUs_Count = min(U_LUs_Count,99)
                    U_IRSs_0_Count = min(U_IRSs_0_Count,99)
                print('done')

        # bootstrap value if not done
        with torch.no_grad():
            next_value_LUs = agent_LUs.get_value(next_obs_LUs).reshape(1, -1)
            advantages_LUs = torch.zeros_like(rewards_LUs).to(device)
            lastgaelam_LUs = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal_LUs = 1.0 - next_done_LUs
                    nextvalues_LUs = next_value_LUs
                else:
                    nextnonterminal_LUs = 1.0 - dones_LUs[t + 1]
                    nextvalues_LUs = values_LUs[t + 1]
                delta_LUs = rewards_LUs[t] + args.gamma * nextvalues_LUs * nextnonterminal_LUs - values_LUs[t]
                advantages_LUs[t] = lastgaelam_LUs = delta_LUs + args.gamma * args.gae_lambda * nextnonterminal_LUs * lastgaelam_LUs
            returns_LUs = advantages_LUs + values_LUs

            next_value_ATK = agent_ATK.get_value(next_obs_ATK).reshape(1, -1)
            advantages_ATK = torch.zeros_like(rewards_ATK).to(device)
            lastgaelam_ATK = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal_ATK = 1.0 - next_done_ATK
                    nextvalues_ATK = next_value_ATK
                else:
                    nextnonterminal_ATK = 1.0 - dones_ATK[t + 1]
                    nextvalues_ATK = values_ATK[t + 1]
                delta_ATK = rewards_ATK[t] + args.gamma * nextvalues_ATK * nextnonterminal_ATK - values_ATK[t]
                advantages_ATK[t] = lastgaelam_ATK = delta_ATK + args.gamma * args.gae_lambda * nextnonterminal_ATK * lastgaelam_ATK
            returns_ATK = advantages_ATK + values_ATK

            next_value_IRSs = agent_IRSs.get_value(next_obs_IRSs).reshape(1, -1)
            advantages_IRSs = torch.zeros_like(rewards_IRSs).to(device)
            lastgaelam_IRSs = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal_IRSs = 1.0 - next_done_IRSs
                    nextvalues_IRSs = next_value_IRSs
                else:
                    nextnonterminal_IRSs = 1.0 - dones_IRSs[t + 1]
                    nextvalues_IRSs = values_IRSs[t + 1]
                delta_IRSs = rewards_IRSs[t] + args.gamma * nextvalues_IRSs * nextnonterminal_IRSs - values_IRSs[t]
                advantages_IRSs[t] = lastgaelam_IRSs = delta_IRSs + args.gamma * args.gae_lambda * nextnonterminal_IRSs * lastgaelam_IRSs
            returns_IRSs = advantages_IRSs + values_IRSs

        # flatten the batch
        b_obs_LUs = obs_LUs.reshape((-1,) + env.observation_space('LUs').shape)
        b_logprobs_LUs = logprobs_LUs.reshape(-1)
        b_actions_LUs = actions_LUs.reshape((-1,) + env.action_space('LUs').shape)
        b_advantages_LUs = advantages_LUs.reshape(-1)
        b_returns_LUs = returns_LUs.reshape(-1)
        b_values_LUs = values_LUs.reshape(-1)

        b_obs_ATK = obs_ATK.reshape((-1,) + env.observation_space('ATK').shape)
        b_logprobs_ATK = logprobs_ATK.reshape(-1)
        b_actions_ATK = actions_ATK.reshape((-1,) + env.action_space('ATK').shape)
        b_advantages_ATK = advantages_ATK.reshape(-1)
        b_returns_ATK = returns_ATK.reshape(-1)
        b_values_ATK = values_ATK.reshape(-1)

        b_obs_IRSs = obs_IRSs.reshape((-1,) + env.observation_space('IRSs').shape)
        b_logprobs_IRSs = logprobs_IRSs.reshape(-1)
        b_actions_IRSs = actions_IRSs.reshape((-1,) + env.action_space('IRSs').shape)
        b_advantages_IRSs = advantages_IRSs.reshape(-1)
        b_returns_IRSs = returns_IRSs.reshape(-1)
        b_values_IRSs = values_IRSs.reshape(-1)


        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs_LUs = []
        clipfracs_ATK = []
        clipfracs_IRSs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # for LUs
                _, newlogprob_LUs, entropy_LUs, newvalue_LUs = agent_LUs.get_action_and_value(b_obs_LUs[mb_inds], b_actions_LUs[mb_inds])
                logratio_LUs = newlogprob_LUs - b_logprobs_LUs[mb_inds]
                ratio_LUs = logratio_LUs.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_LUs = (-logratio_LUs).mean()
                    approx_kl_LUs = ((ratio_LUs - 1) - logratio_LUs).mean()
                    clipfracs_LUs += [((ratio_LUs - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages_LUs = b_advantages_LUs[mb_inds]
                if args.norm_adv:
                    mb_advantages_LUs = (mb_advantages_LUs - mb_advantages_LUs.mean()) / (mb_advantages_LUs.std() + 1e-8)

                # Policy loss
                pg_loss1_LUs = -mb_advantages_LUs * ratio_LUs
                pg_loss2_LUs = -mb_advantages_LUs * torch.clamp(ratio_LUs, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_LUs = torch.max(pg_loss1_LUs, pg_loss2_LUs).mean()

                # Value loss
                newvalue_LUs = newvalue_LUs.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped_LUs = (newvalue_LUs - b_returns_LUs[mb_inds]) ** 2
                    v_clipped_LUs = b_values_LUs[mb_inds] + torch.clamp(
                        newvalue_LUs - b_values_LUs[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped_LUs = (v_clipped_LUs - b_returns_LUs[mb_inds]) ** 2
                    v_loss_max_LUs = torch.max(v_loss_unclipped_LUs, v_loss_clipped_LUs)
                    v_loss_LUs = 0.5 * v_loss_max_LUs.mean()
                else:
                    v_loss_LUs = 0.5 * ((newvalue_LUs - b_returns_LUs[mb_inds]) ** 2).mean()

                entropy_loss_LUs = entropy_LUs.mean()
                loss_LUs = pg_loss_LUs - args.ent_coef * entropy_loss_LUs + v_loss_LUs * args.vf_coef

                optimizer_LUs.zero_grad()
                loss_LUs.backward()
                nn.utils.clip_grad_norm_(agent_LUs.parameters(), args.max_grad_norm)
                optimizer_LUs.step()

                # for ATK
                _, newlogprob_ATK, entropy_ATK, newvalue_ATK = agent_ATK.get_action_and_value(b_obs_ATK[mb_inds], b_actions_ATK[mb_inds])
                logratio_ATK = newlogprob_ATK - b_logprobs_ATK[mb_inds]
                ratio_ATK = logratio_ATK.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_ATK = (-logratio_ATK).mean()
                    approx_kl_ATK = ((ratio_ATK - 1) - logratio_ATK).mean()
                    clipfracs_ATK += [((ratio_ATK - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages_ATK = b_advantages_ATK[mb_inds]
                if args.norm_adv:
                    mb_advantages_ATK = (mb_advantages_ATK - mb_advantages_ATK.mean()) / (mb_advantages_ATK.std() + 1e-8)

                # Policy loss
                pg_loss1_ATK = -mb_advantages_ATK * ratio_ATK
                pg_loss2_ATK = -mb_advantages_ATK * torch.clamp(ratio_ATK, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_ATK = torch.max(pg_loss1_ATK, pg_loss2_ATK).mean()

                # Value loss
                newvalue_ATK = newvalue_ATK.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped_ATK = (newvalue_ATK - b_returns_ATK[mb_inds]) ** 2
                    v_clipped_ATK = b_values_ATK[mb_inds] + torch.clamp(
                        newvalue_ATK - b_values_ATK[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped_ATK = (v_clipped_ATK - b_returns_ATK[mb_inds]) ** 2
                    v_loss_max_ATK = torch.max(v_loss_unclipped_ATK, v_loss_clipped_ATK)
                    v_loss_ATK = 0.5 * v_loss_max_ATK.mean()
                else:
                    v_loss_ATK = 0.5 * ((newvalue_ATK - b_returns_ATK[mb_inds]) ** 2).mean()

                entropy_loss_ATK = entropy_ATK.mean()
                loss_ATK = pg_loss_ATK - args.ent_coef * entropy_loss_ATK + v_loss_ATK * args.vf_coef

                optimizer_ATK.zero_grad()
                loss_ATK.backward()
                nn.utils.clip_grad_norm_(agent_ATK.parameters(), args.max_grad_norm)
                optimizer_ATK.step()

                # for IRSs
                _, newlogprob_IRSs, entropy_IRSs, newvalue_IRSs = agent_IRSs.get_action_and_value(b_obs_IRSs[mb_inds], b_actions_IRSs[mb_inds])
                logratio_IRSs = newlogprob_IRSs - b_logprobs_IRSs[mb_inds]
                ratio_IRSs = logratio_IRSs.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_IRSs = (-logratio_IRSs).mean()
                    approx_kl_IRSs = ((ratio_IRSs - 1) - logratio_IRSs).mean()
                    clipfracs_IRSs += [((ratio_IRSs - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages_IRSs = b_advantages_IRSs[mb_inds]
                if args.norm_adv:
                    mb_advantages_IRSs = (mb_advantages_IRSs - mb_advantages_IRSs.mean()) / (mb_advantages_IRSs.std() + 1e-8)

                # Policy loss
                pg_loss1_IRSs = -mb_advantages_IRSs * ratio_IRSs
                pg_loss2_IRSs = -mb_advantages_IRSs * torch.clamp(ratio_IRSs, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_IRSs = torch.max(pg_loss1_IRSs, pg_loss2_IRSs).mean()

                # Value loss
                newvalue_IRSs = newvalue_IRSs.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped_IRSs = (newvalue_IRSs - b_returns_IRSs[mb_inds]) ** 2
                    v_clipped_IRSs = b_values_IRSs[mb_inds] + torch.clamp(
                        newvalue_IRSs - b_values_IRSs[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped_IRSs = (v_clipped_IRSs - b_returns_IRSs[mb_inds]) ** 2
                    v_loss_max_IRSs = torch.max(v_loss_unclipped_IRSs, v_loss_clipped_IRSs)
                    v_loss_IRSs = 0.5 * v_loss_max_IRSs.mean()
                else:
                    v_loss_IRSs = 0.5 * ((newvalue_IRSs - b_returns_IRSs[mb_inds]) ** 2).mean()

                entropy_loss_IRSs = entropy_IRSs.mean()
                loss_IRSs = pg_loss_IRSs - args.ent_coef * entropy_loss_IRSs + v_loss_IRSs * args.vf_coef

                optimizer_IRSs.zero_grad()
                loss_IRSs.backward()
                nn.utils.clip_grad_norm_(agent_IRSs.parameters(), args.max_grad_norm)
                optimizer_IRSs.step()

        y_pred_LUs, y_true_LUs = b_values_LUs.cpu().numpy(), b_returns_LUs.cpu().numpy()
        var_y_LUs = np.var(y_true_LUs)
        explained_var_LUs = np.nan if var_y_LUs == 0 else 1 - np.var(y_true_LUs - y_pred_LUs) / var_y_LUs

        y_pred_ATK, y_true_ATK = b_values_ATK.cpu().numpy(), b_returns_ATK.cpu().numpy()
        var_y_ATK = np.var(y_true_ATK)
        explained_var_ATK = np.nan if var_y_ATK == 0 else 1 - np.var(y_true_ATK - y_pred_ATK) / var_y_ATK

        y_pred_IRSs, y_true_IRSs = b_values_IRSs.cpu().numpy(), b_returns_IRSs.cpu().numpy()
        var_y_IRSs = np.var(y_true_IRSs)
        explained_var_IRSs = np.nan if var_y_IRSs == 0 else 1 - np.var(y_true_IRSs - y_pred_IRSs) / var_y_IRSs


        writer.add_scalar("Utility/approx_U_LUs", U_LUs, global_step)
        writer.add_scalar("Utility/approx_U_ATK", U_ATK, global_step)
        writer.add_scalar("Utility/approx_U_IRSs_0", U_IRSs_0, global_step)
        writer.add_scalar("Utility/approx_U_IRSs_1", U_IRSs_1, global_step)

        writer.add_scalar("Utility/cru_U_LUs", total_reward_LUs, global_step)
        writer.add_scalar("Utility/cru_U_ATK", total_reward_ATK, global_step)
        writer.add_scalar("Utility/cru_U_IRSs", total_reward_IRSs, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate_LUs", optimizer_LUs.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_LUs", v_loss_LUs.item(), global_step)
        writer.add_scalar("losses/policy_loss_LUs", pg_loss_LUs.item(), global_step)
        writer.add_scalar("losses/entropy_LUs", entropy_loss_LUs.item(), global_step)
        writer.add_scalar("losses/old_approx_kl_LUs", old_approx_kl_LUs.item(), global_step)
        writer.add_scalar("losses/approx_kl_LUs", approx_kl_LUs.item(), global_step)
        writer.add_scalar("losses/clipfrac_LUs", np.mean(clipfracs_LUs), global_step)
        writer.add_scalar("losses/explained_variance_LUs", explained_var_LUs, global_step)

        writer.add_scalar("charts/learning_rate_ATK", optimizer_ATK.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_ATK", v_loss_ATK.item(), global_step)
        writer.add_scalar("losses/policy_loss_ATK", pg_loss_ATK.item(), global_step)
        writer.add_scalar("losses/entropy_ATK", entropy_loss_ATK.item(), global_step)
        writer.add_scalar("losses/old_approx_kl_ATK", old_approx_kl_ATK.item(), global_step)
        writer.add_scalar("losses/approx_kl_ATK", approx_kl_ATK.item(), global_step)
        writer.add_scalar("losses/clipfrac_ATK", np.mean(clipfracs_ATK), global_step)
        writer.add_scalar("losses/explained_variance_ATK", explained_var_ATK, global_step)

        writer.add_scalar("charts/learning_rate_IRSs", optimizer_IRSs.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_IRSs", v_loss_IRSs.item(), global_step)
        writer.add_scalar("losses/policy_loss_IRSs", pg_loss_IRSs.item(), global_step)
        writer.add_scalar("losses/entropy_IRSs", entropy_loss_IRSs.item(), global_step)
        writer.add_scalar("losses/old_approx_kl_IRSs", old_approx_kl_IRSs.item(), global_step)
        writer.add_scalar("losses/approx_kl_IRSs", approx_kl_IRSs.item(), global_step)
        writer.add_scalar("losses/clipfrac_IRSs", np.mean(clipfracs_IRSs), global_step)
        writer.add_scalar("losses/explained_variance_IRSs", explained_var_IRSs, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()
    writer.close()