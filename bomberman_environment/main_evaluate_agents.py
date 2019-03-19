
from time import time, sleep
import contextlib
from time import time

with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
import numpy as np
import multiprocessing as mp
import threading
import os

from environment_updated import BombeRLeWorld, ReplayWorld  # FIXME custom game environment
from settings_agent_evaluation import s  # FIXME custom settings


# Function to run the game logic in a separate thread
def game_logic(world, user_inputs):
    last_update = time()
    while True:
        # Game logic
        if (s.turn_based and len(user_inputs) == 0):
            sleep(0.1)
        elif (s.gui and (time()-last_update < s.update_interval)):
            sleep(s.update_interval - (time() - last_update))
        else:
            last_update = time()
            if world.running:
                try:
                    world.do_step(user_inputs.pop(0) if len(user_inputs) else 'WAIT')
                except Exception as e:
                    world.end_round()
                    raise


def main(evaluate_agents: list, agent_feature_strings: list, savepath:str):
    """
    Run a normal game with a list of custom chosen agents vs. the remaining number of simple agents.
    Save game states to savepath for later review.
    :param evaluate_agents: Test these agents in a game against each other and (4 - len) simple agents
    :param savepath: Save game files here
    :return:
    """
    pygame.init()

    if savepath[-1] not in ("/", "\\"):
        savepath += "/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Emulate Windows process spawning behaviour under Unix (for testing)
    # mp.set_start_method('spawn')

    # Initialize environment and agents

    if len(evaluate_agents) > 4:
        raise RuntimeError("Got too many agents (expected <=4, got", len(evaluate_agents), ")")
    if len(evaluate_agents) != len(agent_feature_strings):
        raise RuntimeError("Mismatch between agents to evaluate and number of feature combinations received.")
    world = BombeRLeWorld(

        [(agent_name, False) for agent_name in evaluate_agents] + [("simple_agent", False) for i in
                                                                   range(4 - len(evaluate_agents))]
            , agent_feature_strings, savepath)

    # world = ReplayWorld('Replay 2019-01-30 16:57:42')
    user_inputs = []

    # Start game logic thread
    t = threading.Thread(target=game_logic, args=(world, user_inputs))
    t.daemon = True
    t.start()

    # Run one or more games
    for i in range(s.n_rounds):
        if not world.running:
            world.ready_for_restart_flag.wait()
            world.ready_for_restart_flag.clear()
            world.new_round()

        # First render
        if s.gui:
            world.render()
            pygame.display.flip()

        round_finished = False
        last_update = time()
        last_frame = time()
        user_inputs.clear()

        # Main game loop
        while not round_finished:
            # Grab events
            key_pressed = None
            for event in pygame.event.get():
                if event.type == QUIT:
                    world.end_round()
                    world.end()
                    return
                elif event.type == KEYDOWN:
                    key_pressed = event.key
                    if key_pressed in (K_q, K_ESCAPE):
                        world.end_round()
                    if not world.running:
                        round_finished = True
                    # Convert keyboard input into actions
                    if s.input_map.get(key_pressed):
                        if s.turn_based:
                            user_inputs.clear()
                        user_inputs.append(s.input_map.get(key_pressed))

            if not world.running and not s.gui:
                round_finished = True

            # Rendering
            if s.gui and (time()-last_frame >= 1/s.fps):
                world.render()
                pygame.display.flip()
                last_frame = time()
            else:
                sleep_time = 1/s.fps - (time() - last_frame)
                if sleep_time > 0:
                    sleep(sleep_time)
                if not s.gui:
                    last_frame = time()

    world.end()


if __name__ == '__main__':
    main()
