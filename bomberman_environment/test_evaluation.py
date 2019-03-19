
import json

from evaluation_environment import EvaluationEnvironment


def main():
    env = EvaluationEnvironment(["testing_only"], "data/games/evaluations")
    env.run_trials()
    json_path = env.analyze_games()[-1]

    with open(json_path, 'r') as f:
        events, durations = json.load(f)

    print()
