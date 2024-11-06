import numpy as np
import pandas as pd
import mdptoolbox
from utils import REGION_LABEL, REGION_CODE

ACTIONS = 3  # 0: maintain 1: increase 2: lower
NUM_STATES_PER_REGION = 3
NUM_STATES = len(REGION_CODE) * NUM_STATES_PER_REGION

def make_transition_prob(df_consumption: pd.DataFrame):
    transition_counts_base = np.zeros((NUM_STATES, NUM_STATES))
    P = np.zeros((ACTIONS, NUM_STATES, NUM_STATES))

    for i in range(1, len(df_consumption)):
        region = df_consumption.iloc[i]["region"]
        region_type = REGION_LABEL[region]
        ts = df_consumption.iloc[i].T[1:]
        ts.index = pd.to_datetime(ts.index)
        ts = ts.astype(float)

        lower_thre = ts.quantile(1 / 3)
        high_thre = ts.quantile(2 / 3)

        df_state = pd.DataFrame(
            {
                "date": ts.index,
                "region": [region_type] * len(ts),
                "consumption": ts.values
            }
        )

        def categorize_state(consumption: float):
            base = REGION_CODE[region_type]
            if consumption < lower_thre:
                return base * 3
            elif consumption > high_thre:
                return base * 3 + 2
            else:
                return base * 3 + 1

        df_state["state"] = df_state["consumption"].apply(categorize_state)

        for k in range(len(df_state) - 1):
            current_state = df_state.iloc[k]['state']
            next_state = df_state.iloc[k + 1]['state']
            transition_counts_base[current_state, next_state] += 1

    P_base = np.zeros((NUM_STATES, NUM_STATES))
    for s in range(NUM_STATES):
        if transition_counts_base[s].sum() > 0:
            P_base[s] = transition_counts_base[s] / transition_counts_base[s].sum()
    P[0] = P_base

    for s in range(NUM_STATES):
        for s_prime in range(NUM_STATES):
            if s_prime == min(s + 1, NUM_STATES - 1):
                P[1, s, s_prime] = min(P_base[s, s_prime] + 0.1, 1)
            else:
                P[1, s, s_prime] = P_base[s, s_prime] * 0.9

            if s_prime == max(s - 1, 0):
                P[2, s, s_prime] = min(P_base[s, s_prime] + 0.1, 1)
            else:
                P[2, s, s_prime] = P_base[s, s_prime] * 0.9

    for a in range(ACTIONS):
        for s in range(NUM_STATES):
            if P[a, s].sum() > 0:
                P[a, s] /= P[a, s].sum()
            else:
                P[a, s] = np.zeros(NUM_STATES)
                P[a, s, s] = 1.0

    return P


def make_rewards():
    R = np.zeros((ACTIONS, NUM_STATES, NUM_STATES))
    for a in range(ACTIONS):
        for s in range(NUM_STATES):
            for s_prime in range(NUM_STATES):
                if s // NUM_STATES_PER_REGION == 0:  # residential
                    if a == 0:
                        R[a, s, s_prime] = 1 if s_prime == s else 0
                    elif a == 1:
                        R[a, s, s_prime] = 2 if s_prime > s else -1
                    elif a == 2:
                        R[a, s, s_prime] = -2 if s_prime < s else 0
                elif s // NUM_STATES_PER_REGION == 1:  # commercial
                    if a == 0:
                        R[a, s, s_prime] = 1
                    elif a == 1:
                        R[a, s, s_prime] = 2
                    elif a == 2:
                        R[a, s, s_prime] = -1
                elif s // NUM_STATES_PER_REGION == 2:  # industrial
                    if a == 0:
                        R[a, s, s_prime] = 1
                    elif a == 1:
                        R[a, s, s_prime] = 2
                    elif a == 2:
                        R[a, s, s_prime] = 0
                elif s // NUM_STATES_PER_REGION == 3:  # suburb
                    if a == 0:
                        R[a, s, s_prime] = 0
                    elif a == 1:
                        R[a, s, s_prime] = -1
                    elif a == 2:
                        R[a, s, s_prime] = 2

    return R


class MDP:

    def __init__(self, P, R, discount):
        self.P = P
        self.R = R
        self.discount = discount
        self.PolicyIteration = mdptoolbox.mdp.PolicyIteration(self.P, self.R, self.discount)

    def get_policy(self):
        self.PolicyIteration.run()
        return self.PolicyIteration.policy
