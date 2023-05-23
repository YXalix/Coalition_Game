import gym
import numpy as np
from gym import spaces
import math

def db2pow(r):
    return np.power(10,r/10)

def pow2db(p):                           
    return 10*np.log10(p)
 
def lossmodel(d,alpha):
    return db2pow(-30)*np.power((d/1),-alpha)


def URA_sv(theta,phi,Nx,Ny):
    m = np.array([i for i in range(Nx)])
    a_az = np.exp(1j*np.pi*m*np.sin(theta)*np.cos(phi))
    a_az = a_az.reshape(-1,1)
    n = np.array([i for i in range(Ny)])
    a_el = np.exp(1j*np.pi*n*np.sin(phi))
    a_el = a_el.reshape(-1,1)
    ura_sv = np.kron(a_az,a_el)
    return ura_sv

def ULA_sv(theta,M):
    m = np.array([i for i in range(M)])
    ula_sv = np.exp(1j*np.pi*m*np.sin(theta)).T
    ula_sv = ula_sv.reshape(-1,1)
    return ula_sv

def whiten(state):
    return (state - np.mean(state)) / np.std(state)

class RIS_MISO(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 num_antennas,
                 num_RIS_elements,
                 num_users,
                 num_eves,
                 power_tdb=30,
                 AWGN_var=1e-8,
                 episode_t = 10000):
        super().__init__()

        self.M = num_antennas
        self.L = num_RIS_elements
        self.Nx = int(math.sqrt(self.L))
        self.Nz = self.Nx
        self.K = num_users
        self.E = num_eves
        self.power_t = db2pow(power_tdb)

        self.awgn_var = AWGN_var

        channel_size = 2*(self.M * self.L + self.M * self.K + self.M * self.E + self.L * self.K + self.L * self.E)
        self.action_dim = self.L + 2 * self.M * self.K
        self.state_dim = channel_size + self.action_dim + 2 + 1

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.state_dim,), dtype=np.float32)

        self.hAU = None
        self.hAE = None
        self.hAI = None
        self.hIU = None
        self.hIE = None
        c0 = None

        self.episode_t = episode_t
        self.episode_num = 0
    
    def setpower(self,t):
        self.power_t = t

    def _compute_reward(self,W,Phi):
        reward = 0

        x = self.hIU @ Phi @ self.hAI @ W + self.hAU @ W

        x = np.abs(x)**2

        y = self.hIE @ Phi @ self.hAI @ W + self.hAE @ W
        y = np.abs(y)**2

        reward = np.log(1+x)/np.log(2) - np.log(1+y)/np.log(2)
        reward = reward.sum()
        Src_rate = reward
        #if np.abs((reward - self.R_sec.item()) / reward) <= 1e-3 :
            #self.done = True
        tramsmitRate = np.log(1+x)/np.log(2)
        tramsmitRate = tramsmitRate.sum()
        
        return reward,tramsmitRate,Src_rate

    def step(self, action):
        self.episode_num += 1

        W_real = action[:self.M*self.K]
        W_imag = action[self.M * self.K:2 * self.M * self.K]

        W = (W_real + 1j * W_imag).reshape(self.M,self.K)
        WW_H = np.matmul(W, np.transpose(W.conj()))
        current_power_t = np.sqrt(np.real(np.trace(WW_H)))/ np.sqrt(self.power_t)
        W = W / current_power_t

        theta = action[-self.L : ]*np.pi
        Phi = np.diag(np.exp(1j*theta).reshape(-1))

        hAE_real , hAE_imag = np.real(self.hAE).flatten() , np.imag(self.hAE).flatten()
        hAU_real , hAU_imag = np.real(self.hAU).flatten() , np.imag(self.hAU).flatten()
        hAI_real , hAI_imag = np.real(self.hAI).flatten() , np.imag(self.hAI).flatten()
        hIE_real , hIE_imag = np.real(self.hIE).flatten() , np.imag(self.hIE).flatten()
        hIU_real , hIU_imag = np.real(self.hIU).flatten() , np.imag(self.hIU).flatten()

        reward,tramsmitRate,Src_rate = self._compute_reward(W,Phi)
        observation = np.hstack((hAE_real,hAE_imag,hAU_real,hAU_imag,hAI_real,hAI_imag,hIE_real,hIE_imag,hIU_real,hIU_imag,action,Src_rate,tramsmitRate,self.power_t))

        observation =  (observation - np.mean(observation)) / np.std(observation)
        
        if self.episode_num >= self.episode_t:
            self.done = True
        return observation.astype(np.float32), reward, self.done, {}

    def reset(self):

        self.episode_t = 0

        self.done = False

        # pass loss parameters
        alpha_AU = 3
        alpha_AE = 3
        alpha_AI = 2.2
        alpha_IU = 2.3
        alpha_IE = 2.5
        c0 = db2pow(-30)

        # rician factors
        k_AU = db2pow(1)
        k_AE = db2pow(1)
        k_AI = db2pow(10)
        k_IU = db2pow(10)
        k_IE = db2pow(10)

        # locations
        # TODO set multi user with different location
        APloc = np.array([0,0]) # AP
        userloc = np.array([150,0]) # user
        eveloc = np.array([145,0]) # eve
        IRSloc = np.array([145,5]) # IRS

        dAE = np.linalg.norm(APloc-eveloc)
        self.hAE = np.sqrt(lossmodel(dAE,alpha_AE)/self.awgn_var)*(np.sqrt(k_AE/(1+k_AE))*np.ones((self.M,self.E)).T.conjugate() + np.sqrt(1/(1+k_AE))*(np.random.randn(self.E,self.M)+1j*np.random.randn(self.E,self.M))/np.sqrt(2))
        
        dAU = np.linalg.norm(APloc-userloc)
        self.hAU = np.sqrt(lossmodel(dAU,alpha_AU)/self.awgn_var)*(np.sqrt(k_AU/(1+k_AU))*np.ones((self.M,self.K)).T.conjugate() + np.sqrt(1/(1+k_AU))*(np.random.randn(self.K,self.M)+1j*np.random.randn(self.K,self.M))/np.sqrt(2))

        dAI = np.linalg.norm(APloc-IRSloc)
        thetaIRS = math.atan(145/5)
        phi = 0
        thetaAP = math.atan(5/145)
        self.hAI = np.sqrt(lossmodel(dAI,alpha_AI)/self.awgn_var)*(np.sqrt(k_AI/(1+k_AI))*URA_sv(thetaIRS,phi,self.Nx,self.Nz)@ULA_sv(thetaAP,self.M).T.conjugate() + np.sqrt(1/(1+k_AI))*(np.random.randn(self.L,self.M)+1j*np.random.randn(self.L,self.M))/np.sqrt(2))
        
        dIU = np.linalg.norm(IRSloc-userloc)
        thetaIRS = -np.pi/4
        phi = 0
        self.hIU = np.sqrt(lossmodel(dIU,alpha_IU))*(np.sqrt(k_IU/(1+k_IU))*URA_sv(thetaIRS,phi,self.Nx,self.Nz).T.conjugate() + np.sqrt(1/(1+k_IU))*(np.random.randn(self.K,self.L)+1j*np.random.randn(self.K,self.L))/np.sqrt(2))


        dIE = np.linalg.norm(IRSloc-eveloc)
        thetaIRS = 0
        phi = 0
        self.hIE = np.sqrt(lossmodel(dIE,alpha_IE))*(np.sqrt(k_IE/(1+k_IE))*URA_sv(thetaIRS,phi,self.Nx,self.Nz).T.conjugate() + np.sqrt(1/(1+k_IE))*(np.random.randn(self.E,self.L)+1j*np.random.randn(self.E,self.L))/np.sqrt(2))

        hAE_real , hAE_imag = np.real(self.hAE).flatten() , np.imag(self.hAE).flatten()
        hAU_real , hAU_imag = np.real(self.hAU).flatten() , np.imag(self.hAU).flatten()
        hAI_real , hAI_imag = np.real(self.hAI).flatten() , np.imag(self.hAI).flatten()
        hIE_real , hIE_imag = np.real(self.hIE).flatten() , np.imag(self.hIE).flatten()
        hIU_real , hIU_imag = np.real(self.hIU).flatten() , np.imag(self.hIU).flatten()

        init_action = np.zeros((self.action_dim,),dtype=np.float32)

        observation = np.hstack((hAE_real,hAE_imag,hAU_real,hAU_imag,hAI_real,hAI_imag,hIE_real,hIE_imag,hIU_real,hIU_imag,init_action,0,0,self.power_t)).astype(np.float32)
        observation =  (observation - np.mean(observation)) / np.std(observation)
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass