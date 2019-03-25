import os
from os import listdir, remove
from os.path import isfile, join
import numpy as np
from datetime import datetime
import datetime
from time import time
import json

import main_evaluate_agents
from agent_code.observation_object import ObservationObject
from state_functions.rewards import event_rewards


class EvaluationEnvironment:
    """
    Provides a collection of methods to run trial games and analyze the results.
    """

    def __init__(self, agent_names:list, save_directory:str):
        """
        Initialize environment attributes.
        :param agent_names: Agent code folder names FIXME (only len == 1 for now)
        :param save_directory: Directory name (created if it doesn't exist)
        """
        self.agent_names = agent_names
        self.save_directory = save_directory if save_directory[-1] not in ["/", "\\"] else save_directory[:-1]

    def run_trials(self, add_folder:bool = False):
        """
        Run a number of test runs and save them to a directory
        :param add_folder If true, create a subdirectory in save_directory before saving games there.
        :return:
        """

        save_dir = self.save_directory[:]
        if add_folder:
            save_dir += "/"+ datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H-%M-%S')
            self.save_directory = save_dir

        main_evaluate_agents.main(self.agent_names, [""], save_dir) # run games and save them to output directory

    def analyze_games(self, destroy_data:bool=False, print_steps_trained_with=None):
        """
        Analyze all games in self.save_directory and save a new .json summary file there.
        :param destroy_data: If True, delete games after analysis
        :param print_steps_trained_with: If not None, print the current number of steps the model under this filepath has trained with.
        :return: event_count_by_game, game_durations, events_path, durations_path
        """
        files = [f for f in listdir(self.save_directory) if isfile(join(self.save_directory, f))]

        event_count_by_game = []  # store event counts here

        game_durations = []

        for file in files:
            if file in ("events.npy", "durations.npy"):
                continue
            eventcount = np.zeros((len(self.agent_names), 17)).astype(int)
            try:
                game = np.load(self.save_directory + "/" + file)
            except OSError:
                print("Skipping " + file + ". Is it a .npy file?")
                continue

            obs = ObservationObject(1, [], None)

            step_num = 0
            for step in game:
                obs.set_state(step)
                playersliving = False

                for player in range(len(self.agent_names)):  # track events for agents of interest
                    if player in obs.living_players or player in obs.just_died:
                        eventcount[player] += obs.events[player]
                        playersliving = True

                step_num += 1
                if not playersliving:
                    break  # all players of interest are dead
            game_durations.append(step_num)
            event_count_by_game.append(eventcount)

        event_count_by_game, game_durations = np.array([ec[0] for ec in event_count_by_game]), np.array(game_durations)
        events_path = self.save_directory + "/" + "events.npy"
        durations_path = self.save_directory + "/" + "durations.npy"

        np.save(events_path, event_count_by_game)
        np.save(durations_path, game_durations)

        if print_steps_trained_with is not None:
            steps_trained_path = self.save_directory + "/" + "current_steps.json"
            with open(steps_trained_path, "w") as f:
                json.dump(self.return_steps_trained_with(print_steps_trained_with), f)

        print("Wrote game info to", events_path, "and", durations_path)

        if destroy_data:  # remove files from disk
            print("Removing game data")
            for file in files:
                remove(self.save_directory+"/"+file)

        return event_count_by_game, game_durations, events_path, durations_path

    def return_steps_trained_with(self, filepath):

        files = [f for f in listdir(filepath) if isfile(join(filepath, f))]

        for file in files:
            if file[:8] == "progress":
                with open(filepath+ "/" + file, 'r') as f:
                    steps, lengths = json.load(f)

                    for i, val in enumerate(steps):
                        if i > 0:
                            steps[i] += steps[i - 1]

                return steps[-1]
        raise RuntimeError("progress.json not found")


    def get_rewards_progress(self, evaluation_folder:str):
        """
        Examines each directory in evaluation folder and notes the results of the evaluations contained therein.

        :param evaluation_folder:
        :return: [[x, y1, y2], ... ] with x being number of steps the model has trained with at time of evaluation,
        y1 being median survival time,
        y2 being median rewards during evaluation at that point
        """

        subdirs = [os.path.join(evaluation_folder, o) for o in os.listdir(evaluation_folder) if os.path.isdir(os.path.join(evaluation_folder,o))]

        ret = []

        for dir in subdirs:

            stepcount_file = dir + "/" + "current_steps.json"
            events_file = dir + "/" + "events.npy"
            durations_file = dir + "/" + "durations.npy"

            with open(stepcount_file, 'r') as f:
                steps_trained = json.load(f)

            events = np.load(events_file)

            durations = np.load(durations_file)

            median_rewards = np.median(np.array([np.sum(player_events * event_rewards) for player_events in events]))
            median_durations = np.median(durations)

            ret.append(np.array([steps_trained, median_durations, median_rewards]))

        ret = np.array(ret)

        sort = np.argsort(ret[:, 0])

        return ret[sort]











