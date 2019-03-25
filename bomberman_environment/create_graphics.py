import os
import json
import numpy as np
from agent_code.observation_object import ObservationObject


os.chdir(os.path.dirname(__file__))


def rewrite_jsons(evaluations_folder:str):
    """
    Intermediate function used to rewrite current_steps.json (originally set to 200, 400, 600 ..)
    :param evaluations_folder Examine all subdirectories here
    :return:
    """

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

    evaluations_folder = "data/qtables/"+obs0.get_file_name_string()+"/evaluations"

    subdirs = [os.path.join(evaluations_folder, o) for o in os.listdir(evaluations_folder) if
               os.path.isdir(os.path.join(evaluations_folder, o))]

    # train_batch = 200
    save_every = 5

    with open("data/qtables/" + obs0.get_file_name_string() + "/progress.json", "r") as f:
        progress, qlen = json.load(f)

    for ind, steps in enumerate(progress):
        if ind != 0:
            progress[ind] += progress[ind - 1]

    for dir in subdirs:
        stepcount_file = dir + "/" + "current_steps.json"

        with open(stepcount_file, "r") as f:

            number_of_files_so_far = json.load(f)

        save_increment_count = number_of_files_so_far//save_every

        with open(stepcount_file, "w") as f:
            json.dump(progress[save_increment_count-1], f)


def get_step_count_from_filename(filename:str):
    """
    Given the name of the last numpy file trained with, return the step count of the model up until that point.
    :param filename: Filename (should end in .npy)
    :return:
    """

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

    progress_path = "data/qtables/"+obs0.get_file_name_string()+"/progress.json"
    records_path = "data/qtables/"+obs0.get_file_name_string()+"/records.json"

    save_every = 5

    with open(progress_path, "r") as f:
        steps, _ = json.load(f)

    with open(records_path, "r") as f:
        names = json.load(f)

    for ind, _ in enumerate(steps):
        if ind != 0:
            steps[ind] += steps[ind - 1]

    target_name_index = None
    for ind, name in enumerate(names):
        if name == filename:
            target_name_index = ind
            break

    if target_name_index is None:
        raise FileNotFoundError(filename)

    target_name_index += 1

    if target_name_index % 5 != 0:
        print("Warning: target name index (starting at 1) is", target_name_index)

    step_index = target_name_index//save_every

    return steps[step_index - 1]



filenames = ["2019-03-19 17-12-25_1574.npy", "2019-03-19 17-12-25_1966.npy", "2019-03-19 17-12-25_2258.npy" ]

step_counts = []

for file in filenames:

    step_counts.append(get_step_count_from_filename(file))

    print("")
# rewrite_jsons("")


