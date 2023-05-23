import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete,Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from ris_miso_utils import *

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["ansi", "rgb_array"],
        "name": "ris_miso_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }
    def __init__(self,    
                 num_BSantennas,
                 num_eves,
                 eves_location,
                 num_EVantennas,
                 num_RIS,
                 ris_location,
                 num_RIS_elements,
                 num_users,
                 users_location,
                 power_tdb=30,
                 power_edb=10,
                 AWGN_var=1e-8,
                 max_cycles = 10000,
                 render_mode=None):
        self.eves_location = eves_location
        self.users_location = users_location
        self.ris_location = ris_location
        self.Ma = num_BSantennas
        self.E = num_eves
        self.Me = num_EVantennas
        self.K = num_RIS
        self.N = num_RIS_elements
        self.Nx = int(np.sqrt(self.N))
        self.Nz = self.Nx
        self.U = num_users

        self.power_a = db2pow(power_tdb)
        self.power_e = db2pow(power_edb)

        self.terminate = False
        self.truncate = False
        self.agents = ["BS", "RISs","EVEs"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_pistons))))

        self._agent_selector = agent_selector(self.agents)

        self.has_reset = False
        self.closed = False
        self.seed()

        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

                             # include beamforming and u_l
        self.action_spaces = {"BS":Box(low=-1, high=1,shape=(self.Ma * self.U * 2 + 1,), dtype=np.float32),
                              # include phases , off/on and x_lr x_er
                              "RISs":Box(low=-1, high=1,shape=(self.K * self.N + self.K + 2,), dtype=np.float32),
                              # include beamforming and u_e
                              "EVEs":Box(low=-1, high=1,shape=(self.Me * self.U * 2 + 1,), dtype=np.float32)}
        
        # part1: bs->users. part2: bs->RIS. part3: bs->EVEs. part4: RISs->users. part5: RISs->EVEs sames to EVEs->RISs. part7: EVEs->users. TODO part8: EVEs->EVEs
        channel_size = 2*(self.Ma * self.U
                           + self.Ma * self.K * self.N
                             + self.Ma * self.E
                               + self.K * self.N * self.U
                                 + self.K * self.N * self.E
                                   + self.Me * self.U
                                    + self.Me * self.E)
        
        self.observation_spaces = {
            "BS":Box(low=-1, high=1,shape=(channel_size #CSI infomation + last cicle(RISs action, EVEs acton and users'SINR) 
                                            + self.action_spaces['RISs']
                                             + self.action_spaces['EVEs']
                                              + self.U,), dtype=np.float32),
            "RISs":Box(low=-1, high=1,shape=(channel_size #CSI infomation + last cicle(BS action, EVEs acton and reward)
                                           + self.action_spaces['BS']
                                             + self.action_spaces['EVEs']
                                              + 1,), dtype=np.float32),
            "EVEs":Box(low=-1, high=1,shape=(channel_size #CSI infomation + last cicle(BS action, RISs acton and EVEs'SINR)
                                           + self.action_spaces['BS']
                                            + self.action_spaces['RISs']
                                             + self.E,), dtype=np.float32)
        }

        self.rewards = {i : 0 for i in self.agents}
        self.terminations = {i : False for i in self.agents}
        self.truncations = {i : False for i in self.agents}
        self.infos = {i: "" for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode

        self.frame = 0
        self.max_cycles = max_cycles

        self.hAU = None
        self.hAE = None
        self.hAI = None
        self.hIU = None
        self.hIE = None
        self.hEI = None # self.hEI = self.hIE
        self.hEU = None
        self.hEE = None

        self.actions = {"BS":self.action_spaces['BS'].sample(),
                              "RISs":self.action_spaces['RISs'].sample(),
                              "EVEs":self.action_spaces['EVEs'].sample()}
        self.last_actions = {"BS":self.action_spaces['BS'].sample(),
                              "RISs":self.action_spaces['RISs'].sample(),
                              "EVEs":self.action_spaces['EVEs'].sample()}

    def seed(self, seed: int) -> None:
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.episode_now = 0

        self.has_reset = True
        self.terminate = False
        self.truncate = False

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()



        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))


        beta_AU = 4
        beta_AE = 4
        beta_AI = 2
        beta_IU = 2
        beta_IE = 2
        beta_EU = 2
        Carrier_frequency = 800 #MHZ

        L0 = -30 #db
        k_AU = 0
        k_AE = 0
        k_AI = 10
        k_IU = 10
        k_IE = 10
        k_EU = 2

        

        APloc = np.array([10,0,10],dtype=np.float32)

        dAU = np.linalg.norm(APloc-self.users_location,axis=1)
        self.hAU = pathloss(L0,beta_AU,dAU)*ricean(k_AU,self.U,self.Ma)

        dAE = np.linalg.norm(APloc-self.eves_location,axis=1)
        self.hAE = pathloss(L0,beta_AE,dAE)*ricean(k_AE,self.E,self.Ma)

        dAI = np.linalg.norm(APloc-self.ris_location,axis=1)
        self.hAI= pathloss(L0,beta_AI,dAI)*ricean(k_AI,self.N,self.Ma,self.K)

        dIU = np.linalg.norm(self.ris_location-self.users_location,axis=1)
        self.hIU = pathloss(L0,beta_IU,dIU) * ricean(k_IU,self.U,self.N,self.K)

        dIE = np.linalg.norm(self.ris_location-self.eves_location,axis=1)
        self.hIE = pathloss(L0,beta_IE,dIE) * ricean(k_IE,self.E,self.N,self.K)

        self.hEI = self.hIE

        dEU = np.linalg.norm(self.eves_location-self.users_location,axis=1)
        self.hEU = pathloss(L0,beta_EU,dEU) * ricean(k_EU,self.U,self.Me,self.E)

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        action = np.asarray(action)
        agent = self.agent_selection

        # set action
        self.actions[agent] = action

        if self._agent_selector.is_last():

            
            self.frame += 1
            pass
        else:
            self._clear_rewards()
        
        self.truncate = self.frame >= self.max_cycles

        if self._agent_selector.is_last():
            self.terminations = dict(
                zip(self.agents, [self.terminate for _ in self.agents])
            )
            self.truncations = dict(
                zip(self.agents, [self.truncate for _ in self.agents])
            )


        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

    def observe(self, agent: str):
        reward = self.rewards[agent]

        return
    
    def observation_space(self, agent):

        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        self.closed = True
    

    def compute_reward(self):
        W_real, W_imag, u_l = self.actions["BS"][:self.Ma*self.U],self.actions["BS"][self.Ma * self.U:2*self.Ma*self.U],self.actions["BS"][-1]
        a,theta,x_lr, x_er = self.actions["RISs"][:self.K], self.actions["RISs"][self.K:self.K * self.N], self.actions["RISs"][-2], self.actions["RISs"][-1]
        F_real, F_imag, u_e = self.actions["EVEs"][:self.Me * self.U], self.actions["EVEs"][self.Me * self.U:2*self.Me * self.U],self.actions["EVEs"][-1]
        
        W = (W_real+1j*W_imag).reshape(self.Ma,self.U)
        WW_H = np.matmul(W,np.transpose(W.conj()))

        current_power_a = np.sqrt(np.real(np.trace(WW_H)))/np.sqrt(self.power_a)
        W = W / current_power_a

        F = (F_real + 1j*F_imag).reshape(self.Me,self.U)
        FF_H = np.matmul(F,np.transpose(F.conj()))
        current_power_e = np.sqrt(np.real(np.trace(FF_H)))/np.sqrt(self.power_e)
        F = F / current_power_e
        theta = theta * np.pi
        theta = theta.reshape(self.K,self.N)
        #Phi = np.diag(np.exp(1j*theta).reshape(-1))

        # compute all parties rewards
        for i in range(self.U):
            signal = self.hAU[i] @ W[:,i]
            
            for k in range(self.K):
                signal += self.hIU[k] @ np.diag(np.exp(1j*theta[k])) @ self.hAI[k] @ W[:,i]
            
        
