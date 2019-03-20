
import json
import numpy as np

from evaluation_environment import EvaluationEnvironment


def main():
    env = EvaluationEnvironment(["testing_only"], "data/games/evaluations", 1)
    env.run_trials()
    _, _, events_path, durations_path = env.analyze_games()

    events = np.load(events_path)
    durations = np.load(durations_path)
    print()


if __name__ == '__main__':
    main()
