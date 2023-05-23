
import gymnasium as gym

from IRSs_world import IRSsWorldEnv

import numpy as np


env = IRSsWorldEnv()



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
while time_slot < 10000:
    C_state_old = C_state
    observations =  env.reset()
    step = 0
    U_LUs_IRSs = 2000
    U_ATK_IRSs = 2000
    while step < 100:
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
        # stage 2 decision making
        actions = {}
        actions['LUs'] = env.action_spaces['LUs'].sample()
        actions['IRSs'] = env.action_spaces['IRSs'].sample()
        actions['ATK'] = env.action_spaces['ATK'].sample()

        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        step += 1
        # update coalition utilities information
        U_LUs_IRSs = infos['utilities'][4]
        U_ATK_IRSs = infos['utilities'][5]
        print(C_state)
        print(U_LUs_IRSs)
        print(U_ATK_IRSs)
        print("****************")
        done = terminateds['LUs']
        if done or step == 100:  
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
                U_LUs = U_LUs * U_LUs_Count / (U_LUs_Count + 1) + infos[0]/((U_LUs_Count + 1))
                U_IRSs_0 = U_IRSs_0*U_IRSs_0_Count/(U_IRSs_0_Count+1) + infos['IRSs']/(U_IRSs_0_Count+1)
                U_ATK_IRSs_Count = min(U_ATK_IRSs_Count,5)
                U_LUs_Count = min(U_LUs_Count,99)
                U_IRSs_0_Count = min(U_IRSs_0_Count,99)
            break
    

            """
            if C_state == 0:
                U_LUs = U_LUs*U_LUs_Count/(U_LUs_Count+1) + infos['LUs']*1/(U_LUs_Count+1)
                U_ATK = U_ATK*U_ATK_Count/(U_ATK_Count+1) + infos['ATK']*1/(U_ATK_Count+1)
                U_IRSs_0 = U_IRSs_0*U_IRSs_0_Count/(U_IRSs_0_Count+1) + infos['IRSs']*1/(U_IRSs_0_Count+1)
                U_IRSs_1 = U_IRSs_1*U_IRSs_1_Count/(U_IRSs_1_Count+1) + infos['IRSs']*1/(U_IRSs_1_Count+1)
                U_LUs_Count = min(U_LUs_Count,99)
                U_ATK_Count = min(U_ATK_Count,99)
                U_IRSs_0_Count = min(U_IRSs_0_Count,99)
                U_IRSs_1_Count = min(U_IRSs_1_Count,99)
            elif C_state == 1:
                U_LUs_IRSs = infos['LUs']
                U_ATK = U_ATK*U_ATK_Count/(U_ATK_Count+1) + infos['ATK']*1/(U_ATK_Count+1)
                U_IRSs_1 = U_IRSs_1*U_IRSs_1_Count/(U_IRSs_1_Count+1) + infos['IRSs']*1/(U_IRSs_1_Count+1)
                U_LUs_IRSs_Count = min(U_LUs_IRSs_Count,5)
                U_ATK_Count = min(U_ATK_Count,99)
                U_IRSs_1_Count = min(U_IRSs_1_Count,99)

            elif C_state == 2:
                U_ATK_IRSs = U_ATK_IRSs*U_ATK_IRSs_Count/(U_ATK_IRSs_Count+1) + infos['ATK']*1/(U_ATK_IRSs_Count+1)
                U_IRSs_1 = U_IRSs_1*U_IRSs_1_Count/(U_IRSs_1_Count+1) + infos['IRSs']*1/(U_IRSs_1_Count+1)
                U_IRSs_0 = U_IRSs_0*U_IRSs_0_Count/(U_IRSs_0_Count+1) + infos['IRSs']*1/(U_IRSs_0_Count+1)
                U_ATK_IRSs_Count = min(U_ATK_IRSs_Count,5)
                U_LUs_Count = min(U_LUs_Count,99)
                U_IRSs_0_Count = min(U_IRSs_0_Count,99)
            """