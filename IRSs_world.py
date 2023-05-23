import gymnasium as gym
from gymnasium import Space
from pettingzoo.utils.env import AgentID
import pygame
import numpy as np

from pettingzoo import ParallelEnv

from gymnasium.spaces import Box

from utils import *


class IRSsWorldEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 4,
        "is_parallelizable": True
        }

    def __init__(self,                  
                 BS_antennas = 4,
                 ATK_location = np.array([[10,60,5]]),
                 ATK_antennas = 4,
                 num_RIS = 2,
                 IRSs_location = np.array([[5,67,2],[27,62,2]]),
                 num_RIS_elements = 16,
                 num_users = 2,
                 LUs_location = np.array([[30,62,2],[8,67,2]]),
                 power_bsdb=30,
                 power_atkdb=10,
                 max_cycles = 10,
                 render_mode=None):
        self.Carrier_frequency = 800e6 # Hz
        self.BS_location = np.array([[0,0,10]])
        self.Ma = BS_antennas
        self.ATK_location = ATK_location
        self.Me = ATK_antennas
        self.K = num_RIS
        self.IRSs_location = IRSs_location
        self.N = num_RIS_elements

        self.L = num_users
        self.LUs_location = LUs_location
        self.Ny = int(np.sqrt(self.N))
        self.Nz = self.Ny
        self.power_bs = db2pow(power_bsdb)
        self.power_ATK = db2pow(power_atkdb)
        self.AWGN_var = db2pow(-160)
        self.ISR_var = db2pow(-160)
        self.max_cycles = max_cycles
        self.render_mode = render_mode

        self.terminate = False
        self.truncate = False
        self.now = 0

        self.has_reset = False
        self.closed = False
        self.seed(1)

        self.agents = ["LUs", "IRSs","ATK"]
        self.possible_agents = self.agents[:]

        # part1: bs->users. part2: bs->IRSs. part3: bs->EVEs. part4: IRSs->users. part5: IRSs->ATK part6: ATK->users
        channel_size = 2*(self.Ma * self.L
                           + self.Ma * self.K * self.N
                             + self.Ma
                               + self.K * self.N * self.L
                                 + self.K * self.N
                                   + self.Me * self.L)
        
        self.observation_spaces = {
            # include channel infomation state and U_LUs, U_LUs_IRSs, U_IRSs_0 , C_state_old, C_state
            "LUs":Box(low=-np.inf, high=np.inf,shape=(channel_size+ 6,), dtype=np.float32), 
            # include channel infomation state and U_IRSs_0, U_IRSs_1, U_LUs_IRSs, U_ATK_IRSs, U_LUs, U_ATK, C_state_old, C_state
            "IRSs":Box(low=-np.inf, high=np.inf,shape=(channel_size+ 9,), dtype=np.float32),
            # include channel infomation state and U_ATK, U_ATK_IRSs, U_IRSs_0 , C_state_old, C_state
            "ATK":Box(low=-np.inf, high=np.inf,shape=(channel_size + 6,), dtype=np.float32)
        }
        self.action_spaces = {
            # include beamforming and power allocation for each user
            "LUs":Box(low=-1, high=1,shape=(self.Ma * self.L * 2 + self.L,), dtype=np.float32),
             # include phases and whether active or not kth IRS
            "IRSs":Box(low=-1, high=1,shape=(self.K * self.N + self.K,), dtype=np.float32),
            # include beamforming, power allocation for each user, total power allocation rate
            "ATK":Box(low=-1, high=1,shape=(self.Me * self.L * 2 + self.L + 1,), dtype=np.float32)}

    # utilities = [U_LUs,U_ATK,U_IRSs_0,U_IRSs_1,U_LUs_IRSs,U_ATK_IRSs]
    def set_utilities(self, utilities):
        self.utilities = utilities

    # C_state
    # 0: LUs, ATK, IRSs
    # 1: {LUs, IRSs}, ATK
    # 2: LUs, {ATK, IRSs}
    def set_coalition(self,C_state_old, C_state):
        self.C_state = C_state
        self.C_state_old = C_state_old

    def set_coalition_decisions(self,coalition_decisions):
        self.coalition_decisions = coalition_decisions
    
    def seed(self, seed: int) -> None:
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if seed is not None:
            self.seed(seed)
        # TODO generate the channel state information
        self.generate_channel()
        observations = {}
        observations['LUs'] =  self.observation_spaces['LUs'].sample()
        observations['IRSs'] =  self.observation_spaces['IRSs'].sample()
        observations['ATK'] =  self.observation_spaces['ATK'].sample()

        self.now = 0
        return observations, {}

    def step(self, actions):
        LUs_actions = actions['LUs'].flatten()
        IRSs_actions = actions['IRSs'].flatten()
        ATK_actions = actions['ATK'].flatten()
        
        self.Ws = []
        self.Fs = []
        self.Phis = []
        self.actives = []


        self.LUs_power_allocation_rates = np.array([(LUs_actions[i * (self.Ma * 2 + 1) + self.Ma * 2]+1)/2 for i in range(self.L)])
        ATK_rate = [(ATK_actions[i * (self.Me * 2 + 1) + self.Me * 2]+1)/2 for i in range(self.L)]
        ATK_rate_sum = sum(ATK_rate)
        self.ATK_total_power_allocation_rate = (ATK_actions[-1]+1) / 2
        for i in range(self.L):
            index_LUs = i * (self.Ma * 2 + 1)
            W_real = LUs_actions[index_LUs : index_LUs + self.Ma]
            W_imag = LUs_actions[index_LUs + self.Ma : index_LUs + self.Ma*2]
            power_rate = (LUs_actions[index_LUs + self.Ma*2 : index_LUs + self.Ma*2 + 1] + 1) / 2
            W = (W_real + 1j * W_imag).reshape(self.Ma,1)
            WW_H = np.matmul(W, np.transpose(W.conj()))
            current_power_t = np.sqrt(np.real(np.trace(WW_H))) / (np.sqrt(self.power_bs)*power_rate)
            W = W / current_power_t

            self.Ws.append(W)

            index_ATK = i * (self.Me * 2 + 1)
            F_real = ATK_actions[index_ATK : index_ATK + self.Me]
            F_imag = ATK_actions[index_ATK + self.Me : index_ATK + self.Me * 2]

            # ATK use sum power rate, not each
            power_rate = ATK_rate[i] / ATK_rate_sum
            F = (F_real + 1j * F_imag).reshape(self.Me,1)
            FF_H = np.matmul(F,np.transpose(F.conj()))
            current_power_t = np.sqrt(np.real(np.trace(FF_H))) / (np.sqrt(self.power_ATK)*power_rate * self.ATK_total_power_allocation_rate)
            F = F / current_power_t

            self.Fs.append(F)

        for k in range(self.K):
            index_IRSs = self.N + 1
            theta = IRSs_actions[index_IRSs : index_IRSs + self.N] * np.pi
            Phi = np.diag(np.exp(1j*theta).reshape(-1))
            active = 1 if IRSs_actions[index_IRSs + self.N : index_IRSs + self.N + 1] > 0 else 0
            
            self.Phis.append(Phi)
            self.actives.append(active)

        self.actives = np.array(self.actives)
        H_a_alli = [self.get_H_ai(i) for i in range(self.L)]
        H_a_e = self.get_H_ae()

        H_e_alli = self.H_e_alli

        rewards, infos = self._compute_rewards_info(H_a_alli, H_a_e, H_e_alli)

        self.now += 1
        if self.now >= 200:
            terminateds = True
            truncateds = True
            observations, _ = self.reset()
            return observations, rewards, terminateds, truncateds, infos
        else:
            terminateds = False
            truncateds = False
            
            observations = {}

            HAK_real, HAK_imag =  np.real(self.G_a_allk).flatten(), np.imag(self.G_a_allk).flatten()
            HAL_real, HAL_imag = np.real(self.H_a_alli).flatten(), np.imag(self.H_a_alli).flatten()
            HAE_real, HAE_imag = np.real(self.H_a_alle).flatten(), np.imag(self.H_a_alle).flatten()
            HEI_real, HEI_imag = np.real(self.H_e_alli).flatten(), np.imag(self.H_e_alli).flatten()
            HKI_real, HKI_imag = np.real(self.g_allk_alli).flatten(), np.imag(self.g_allk_alli).flatten()
            HKE_real, HKE_imag = np.real(self.g_allk_alle).flatten(), np.imag(self.g_allk_alle).flatten()

            channels = np.hstack((HAK_real, HAK_imag,HAL_real, HAL_imag,HAE_real, HAE_imag,HEI_real, HEI_imag,HKI_real, HKI_imag,HKE_real, HKE_imag))
            observations['LUs'] = np.hstack((rewards['LUs'], infos['utilities'][0],infos['utilities'][4],infos['utilities'][2],self.C_state_old,self.C_state,channels))
            observations['LUs'] =  (observations['LUs'] - np.mean(observations['LUs'])) / np.std(observations['LUs'])

            observations['ATK'] = np.hstack((rewards['ATK'], infos['utilities'][1],infos['utilities'][5],infos['utilities'][3],self.C_state_old,self.C_state,channels))
            observations['ATK'] =  (observations['ATK'] - np.mean(observations['ATK'])) / np.std(observations['ATK'])
            
            observations['IRSs'] = np.hstack((rewards['IRSs'], infos['utilities'],self.C_state_old,self.C_state,channels))
            observations['IRSs'] =  (observations['IRSs'] - np.mean(observations['IRSs'])) / np.std(observations['IRSs'])

            return observations, rewards, terminateds, truncateds, infos


    def get_H_ai(self,i):

        H_ai = self.H_a_alli[i]
        for k in range(self.K):
            if self.actives[k] == 1:
                H_ai += self.g_allk_alli[k][i] @ self.Phis[k] @ self.G_a_allk[k]
        return H_ai

    def get_H_ae(self):

        H_ae = self.H_a_alle[0]
        for k in range(self.K):
            if self.actives[k] == 1:
                H_ae += self.g_allk_alle[k] @ self.Phis[k] @ self.G_a_allk[k]
        return H_ae
    

    def _compute_rewards_info(self, H_a_alli, H_a_e, H_e_alli):
        # TODO compute the rewards
        SINR_LUs = np.array([self.compute_SINR_LUs_i(i,H_a_alli[i], H_e_alli[i]).item() for i in range(self.L)])
        SINR_ATK = np.array([self.compute_SINR_ATK_i(i,H_a_e).item() for i in range(self.L)])
        

        zeros = np.zeros_like(SINR_LUs)

        V_LUs = np.maximum(np.log2(1+SINR_LUs) - np.log2(1+SINR_ATK),zeros)
        V_LUs = np.sum(V_LUs)

        V_ATK = np.minimum(np.log2(1+SINR_ATK)-np.log2(1+SINR_LUs),zeros)
        V_ATK = np.sum(V_ATK)
        
        LUs_epsilon = 1.2
        ATK_epsilon = 1.2
        IRSs_epsilon = 1.2

        circle_power_cost = db2pow(10)
        BS_circle_power_cost = db2pow(10)
        U_LUs_0 = V_LUs# - LUs_epsilon * np.sum(self.LUs_power_allocation_rates * self.power_bs) - circle_power_cost*(self.L) - BS_circle_power_cost
        U_ATK_0 = V_ATK# - ATK_epsilon * self.ATK_total_power_allocation_rate * self.power_ATK - circle_power_cost
        U_IRSs_0 = 0.0#- IRSs_epsilon * np.sum(self.actives) * self.N * circle_power_cost


        rewards = {'LUs':0.0,'ATK':0.0,'IRSs':0.0}
        infos = {'LUs':0.0,'ATK':0.0,'IRSs':0.0}
        infos['utilities'] = self.utilities

        # compute shapley value for
        if self.C_state == 0:
            infos['LUs'] = U_LUs_0
            infos['ATK'] = U_ATK_0
            infos['IRSs'] = U_IRSs_0

        elif self.C_state == 1:
            infos['utilities'][4] = U_LUs_0 + U_IRSs_0
            infos['LUs'] = (infos['utilities'][4] + self.utilities[0] - self.utilities[2])/2
            infos['ATK'] = U_ATK_0
            infos['IRSs'] = (infos['utilities'][4] + self.utilities[2] - self.utilities[0])/2

            
        elif self.C_state == 2:
            infos['utilities'][5] = U_ATK_0 + U_IRSs_0
            infos['LUs'] = U_LUs_0
            infos['ATK'] = (infos['utilities'][5] + self.utilities[1] - self.utilities[3])/2
            infos['IRSs'] = (infos['utilities'][5] + self.utilities[3] - self.utilities[1])/2
      
        rewards['LUs'] = infos['LUs']
        rewards['ATK'] = infos['ATK']
        rewards['IRSs'] = infos['IRSs']
        # appand punishment
        if self.C_state_old == 1 and self.C_state != 1:
            if self.coalition_decisions[0] == 0:
                rewards['LUs'] = rewards['LUs'] - 0.2*abs(rewards['LUs'])
            if self.coalition_decisions[2] == 0:
                rewards['IRSs'] = rewards['IRSs'] - 0.2*abs(rewards['IRSs'])
        elif self.C_state_old == 2 and self.C_state != 2:
            if self.coalition_decisions[1] == 0:
                rewards['ATK'] = rewards['ATK'] - 0.2*abs(rewards['ATK'])
            if self.coalition_decisions[3] == 0:
                rewards['IRSs'] = rewards['IRSs'] - 0.2*abs(rewards['IRSs'])

        return rewards, infos
        
    def compute_SINR_LUs_i(self,i,H_a_i, H_e_i):

        x = H_a_i @ self.Ws[i]

        x = np.abs(x) ** 2

        y = self.AWGN_var
        for j in range(self.L):
            if j != i:
                y += np.abs(H_a_i @ self.Ws[j])**2
            y += np.abs(H_e_i @ self.Fs[j])**2

        return x / y

    def compute_SINR_ATK_i(self,i, H_a_e):

        x = H_a_e @ self.Ws[i]
        x = np.abs(x) ** 2

        y = self.AWGN_var + self.ISR_var
        for j in range(self.L):
            if j != i:
                y += np.abs(H_a_e @ self.Ws[j])**2
        
        return x / y


    def generate_channel(self):
        # TODO generate the channel state information (CSI) ricean fading channel but now is rayleigh fading channel
        L0 = -30 #db
        beta_ai = 4
        beta_ae = 4
        beta_ak = 2
        beta_ki = 2
        beta_ke = 2
        beta_ei = 2

        distance = lambda x,y: np.linalg.norm(x-y,axis=1)
        dis_ak = distance(self.BS_location,self.IRSs_location)

        self.G_a_allk = [pathloss(L0,beta_ak,dis_ak[k]) * rayleigh(self.Ma,self.N) for k in range(self.K)]
        
        dis_ai = distance(self.BS_location,self.LUs_location)
        self.H_a_alli = [pathloss(L0,beta_ai,dis_ai[i]) * rayleigh(self.Ma,1) for i in range(self.L)]
        
        dis_ae = distance(self.BS_location,self.ATK_location)
        self.H_a_alle = [pathloss(L0,beta_ae,dis_ae) * rayleigh(self.Ma,1)]

        dis_ei = distance(self.ATK_location,self.LUs_location)
        self.H_e_alli = [pathloss(L0,beta_ei,dis_ei[i]) * rayleigh(self.Me,1) for i in range(self.L)]
        
        dis_ki = [distance(self.IRSs_location[k],self.LUs_location) for k in range(self.K)]
        self.g_allk_alli = []
        for k in range(self.K):
            self.g_allk_alli.append([pathloss(L0,beta_ki,dis_ki[k][i]) * rayleigh(self.N,1) for i in range(self.L)])

        dis_ke = distance(self.IRSs_location,self.ATK_location)
        self.g_allk_alle = [pathloss(L0,beta_ke,dis_ke[k]) * rayleigh(self.N,1) for k in range(self.K)]


    def _get_terminal(self):
        return {'ATK':self.terminate,'IRSs':self.terminate,'LUs':self.terminate}

    def _get_tuncate(self):
        return {'ATK':self.truncate,'IRSs':self.truncate,'LUs':self.truncate}

    def action_space(self, agent: AgentID) -> Space:
        return self.action_spaces[agent]
    

    def observation_space(self, agent: AgentID) -> Space:
        return self.observation_spaces[agent]


    def render(self):
        print("render")

    def close(self):
        print("close")
