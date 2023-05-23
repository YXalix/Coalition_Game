import numpy as np


def db2pow(r):
    return np.power(10,r/10)

def pow2db(p):
    return 10*np.log10(p)

def pathloss(L0,beta,distance):
    L0 = db2pow(L0)
    return np.sqrt(L0*np.power(distance,-beta))

def rayleigh(transmitter_num,receiver_num):
    return (np.random.randn(receiver_num,transmitter_num) + 1j * np.random.randn(receiver_num,transmitter_num)) * np.sqrt(0.5)

def ricean(k_factor,receiver_num,transmitter_num,num = None):
        
    #TODO consider AoD and AoA
    LoS_Part = np.sqrt(k_factor/(k_factor+1))*np.ones((receiver_num,transmitter_num, num))

    NLoS_Part = np.sqrt(1/(k_factor+1))*rayleigh(receiver_num,transmitter_num, num)

    return LoS_Part + NLoS_Part


if __name__=="__main__":
    aploc = np.array([[10,0,10]])

    risloc = np.array([[5,57,2],[30,62,2]])

    d = np.linalg.norm(aploc-risloc,axis=1)

    print(d)