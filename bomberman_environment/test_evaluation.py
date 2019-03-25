
import json
import numpy as np
from state_functions.rewards import event_rewards
import matplotlib.pyplot as plt

from evaluation_environment import EvaluationEnvironment
from agent_code.observation_object import ObservationObject

obs0 = ObservationObject(0, ['d_closest_coin_dir',
                             'd_closest_safe_field_dir',
                             'me_has_bomb',
                             'dead_end_detect',
                             'd4_is_safe_to_move_a_l',
                             'd4_is_safe_to_move_b_r',
                             'd4_is_safe_to_move_c_u',
                             'd4_is_safe_to_move_d_d',
                             'd_best_bomb_dropping_dir',
                             'd_closest_enemy_dir'
                             ], None)
def main():

    env = EvaluationEnvironment(["testing_only"], "data/qtables/"+obs0.get_file_name_string()+"/reg_evals")
    env.run_trials(add_folder=True)
    _, _, events_path, durations_path = env.analyze_games()

    events = np.load(events_path)
    durations = np.load(durations_path)

    rewards = [np.sum(player_events * event_rewards ) for player_events in events]

def show_rewards_progress():
    """

    :return:
    """
    env = EvaluationEnvironment(["testing_only"], "data/games/SELFPLAYr1_ismal_ismbr_ismcu_ismdd_ccdir_ccrdir_csfdir_ded_mhb/ITERATION_1")

    q_performance = env.get_rewards_progress("data/qtables/"+obs0.get_file_name_string() + "/evaluations")
    class_reg_performance = env.get_rewards_progress("data/qtables/"+obs0.get_file_name_string() + "/reg_evals")

    class_performance = class_reg_performance[::2]
    reg_performance = class_reg_performance[1:][::2]

    labelsize = 15

    plt.rcParams.update({'font.size': labelsize})
    qcolor = "blue"
    classcolor = "green"
    regcolor = "yellow"

    fig, ax1 = plt.subplots()
    ax1.plot(q_performance[:, 0], q_performance[:, 2], color=qcolor, label="q")
    ax1.set_xlabel("Number of steps seen")
    ax1.set_ylabel("Median reward earned in 7 trial games")

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #ax2.set_ylabel("Classification/Regression", color="black")
    ax1.plot(class_performance[:, 0], class_performance[:, 2], c=classcolor, label="class.")
    ax1.plot(reg_performance[:, 0], reg_performance[:, 2], c=regcolor, label="reg.")

    ax1.xaxis.label.set_fontsize(labelsize)
    ax1.yaxis.label.set_fontsize(labelsize)
    ax1.xaxis.offsetText.set_fontsize(labelsize)
    ax1.yaxis.offsetText.set_fontsize(labelsize)

    title = "Performance comparison of classification, regression, and Q Tables"
    fig.suptitle(title)
    #plt.xlabel("Number of steps seen")
    #plt.ylabel("Median reward earned in 7 trial games")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.legend(loc="upper left")
    fig.tight_layout()
    plt.show()

    print()


if __name__ == '__main__':
    #main()
    show_rewards_progress()


