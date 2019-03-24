
import json
import numpy as np
from state_functions.rewards import event_rewards
import matplotlib.pyplot as plt

from evaluation_environment import EvaluationEnvironment


def main():
    env = EvaluationEnvironment(["testing_only"], "data/games/SELFPLAYr1_ismal_ismbr_ismcu_ismdd_ccdir_ccrdir_csfdir_ded_mhb/ITERATION_1")
    env.run_trials(add_folder=True)
    _, _, events_path, durations_path = env.analyze_games()

    events = np.load(events_path)
    durations = np.load(durations_path)

    rewards = [np.sum(player_events * event_rewards ) for player_events in events]

    ret = env.get_rewards_progress("data/qtables/r0_ismal_ismbr_ismcu_ismdd_bbdd_ccdir_csfdir_ded_mhb/evaluategames2019-03-24 01-17-40")

    plt.plot(ret[:, 0], ret[:, 2])

    plt.show()

    print()


if __name__ == '__main__':
    main()
