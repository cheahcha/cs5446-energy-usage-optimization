from collections import defaultdict
import os
import pandas as pd


from Predictor import Predictor
import MDP
from constants import REGION_LABEL, REGION_CODE_INVERSE, POLICY_MAP, REGION_CODE

# Set toggles
HOLIDAY_FLAG = True
DISCOUNT_FACTOR = 0.6
NEXT_FORECAST_PERIOD = "2024-07"
TO_SAVE = True

def vote(region: str, code: int, policy: tuple) -> defaultdict:
    """
    Vote for optimal policy based on region's sector breakdown.

    Args:
        region (str): Region name
        code (int): State code which indicates whether the demand level is low, medium or high
        policy (tuple): Policies after policy iteration

    Returns:
        defaultdict: Action Map
    """
    region_break = REGION_LABEL[region]
    power_demand = code % 3
    action_map = defaultdict(float)
    for region_type in region_break:
        action_code = policy[REGION_CODE[region_type] * 3 + power_demand]
        print(f"\tRegion type: {region_type} -> {POLICY_MAP[action_code]}")
        action_map[action_code] += region_break[region_type]
    return action_map


if __name__ == "__main__":
    # Load data
    print("Loading electricity consumption data ...")
    data = pd.read_csv('./data/region_monthly_electricity_consumption.csv').rename(columns={"Unnamed: 0": "region"})

    # Training time series model
    print(f"Training prophet model ...")
    predictor = Predictor(data)

    # Make transition probability
    print("Setting up MDP ...")
    P = MDP.make_transition_prob(data)
    print(f"Transition Probability Matrix (P) shape: {P.shape}")

    if HOLIDAY_FLAG:
        R = MDP.make_rewards("holiday")
    else:
        R = MDP.make_rewards("common")
    print(f"Rewards Matrix (R) shape: {R.shape}")

    mdp = MDP.MDP(P, R, DISCOUNT_FACTOR)
    policy = mdp.get_policy()

    print(f"Time series forecasting period: {NEXT_FORECAST_PERIOD}")

    policy_result = pd.DataFrame(columns=['region', 
                                          'sector_breakdown', 
                                          'maintain_action', 
                                          'increase_action', 
                                          'decrease_action',
                                          'optimal_action'
                                          ])
    
    for region in REGION_LABEL:
        
        code, y_pred = predictor.predict(region, NEXT_FORECAST_PERIOD)
        dominant_region_type = REGION_CODE_INVERSE[code // 3]
        action_map = vote(region, code, policy)
        optimal_action_code = max(action_map, key=action_map.get)
        optimal_action = POLICY_MAP[optimal_action_code]

        print(f"{region} - {dominant_region_type}: {optimal_action}")
        print("========")

        policy_result.loc[len(policy_result)] = {
            'region': region,
            'sector_breakdown': REGION_LABEL[region],
            'maintain_action': action_map[0],
            'increase_action': action_map[1],
            'decrease_action': action_map[2],
            'optimal_action': optimal_action
        }
    
    print(f"Policy result df:\n {policy_result.tail()}")

    if TO_SAVE:
        if not os.path.exists('output'):
            os.mkdir('output')

        policy_result.to_csv(f'output/region_policy_result_{NEXT_FORECAST_PERIOD}.csv')
