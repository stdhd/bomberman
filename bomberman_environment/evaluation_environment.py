from os import listdir, remove
from os.path import isfile, join
import numpy as np
import json

import main_evaluate_agents
from agent_code.observation_object import ObservationObject


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

    def run_trials(self):
        """
        Run a number of test runs and save them to a directory
        :return:
        """

        main_evaluate_agents.main(self.agent_names, [""], self.save_directory) # run games and save them to output directory

    def analyze_games(self, destroy_data:bool=False):
        """
        Analyze all games in a directory and save a new .json summary file there.
        :param destroy_data: If yes, delete games after analysis
        :return:
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

        print("Wrote game info to", events_path, "and", durations_path)

        if destroy_data:  # remove files from disk
            print("Removing game data")
            for file in files:
                remove(self.save_directory+"/"+file)

        return event_count_by_game, game_durations, events_path, durations_path


