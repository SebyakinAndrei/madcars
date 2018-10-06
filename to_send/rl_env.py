from mechanic.strategy import KeyboardClient, FileClient, Client
from mechanic.game import Game
import asyncio
import traceback
import json
import numpy as np
from timeit import default_timer

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tensorforce.agents import DQFDAgent
from tensorforce.execution import Runner

import sys

from queue import Queue
from threading import Thread
from my_strategy import RLEnv
import multiprocessing as mp

backend = 'ddqn' # 'dqfd'


def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode,
                                                                                 ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


def run_strategy(msg_in, msg_out, model_q, msg_q, debug_file=None, debug=None, load_weights=None, train=False,
                 episodes=10000, reset_every=1000, self_play=False, second=False, weights_path=None, dueling=False):
    if not self_play:
        reset_every = episodes
    if weights_path is None:
        weights_path = 'weights'
    debug_file = open(weights_path + '/' + debug_file, 'w') if debug_file is not None else None
    if episodes % reset_every != 0:
        raise Exception('episodes {} cant be divided by reset_every {}'.format(episodes, reset_every))
    # Get the environment and extract the number of actions.
    #env = self.env
    env = RLEnv(msg_in, msg_out, msg_q, debug=debug, debug_file=debug_file, second=second)
    np.random.seed(123)
    nb_actions = 3

    model, dqn = None, None
    if backend == 'ddqn':
        model = Sequential()
        model.add(Flatten(input_shape=(1, 16)))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions, activation='linear'))
        # print(model.summary(), file=env.debug_file)

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=1)
        policy = BoltzmannQPolicy()
        # enable the dueling network
        # you can specify the dueling_type to one of {'avg','max','naive'}
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                       enable_dueling_network=dueling, dueling_type='avg', target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if load_weights is not None:
            path = '{}/duel_dqn_{}.h5f'.format(weights_path, load_weights)
            dqn.load_weights(path)
            if debug['general']:
                print('Weights loaded! {}'.format(path), file=env.debug_file, flush=True)
    else:
        network_spec = [
            # dict(type='embedding', indices=100, size=32),
            # dict(type'flatten'),
            dict(type='dense', size=32, activation='relu'),
            dict(type='dense', size=32, activation='relu'),
            dict(type='dense', size=32, activation='relu')
        ]
        agent = DQFDAgent(
            states=dict(shape=(1, 16), type='float'),
            actions=dict(num_actions=3, type='int'),
            network=network_spec,
            # Agent
            states_preprocessing=None,
            actions_exploration=None,
            reward_preprocessing=None,
            discount=0.97,
            optimizer={
                'type': 'adam',
                'learning_rate': 1e-3
            },
            demo_memory_capacity=10000,
            demo_sampling_ratio=0.5,
            # MemoryModel
            update_mode={
                "unit": "timesteps",
                "batch_size": 64,
                "frequency": 4
            },
            memory={
                "type": "prioritized_replay",
                "capacity": 100000,
                "include_next_states": True
            },
            # DistributionModel
            distributions=None,
            entropy_regularization=None,
            execution=dict(
                type='single',
                session_config=None,
                distributed_spec=None
            )
        )

        # Create the runner
        dqn = Runner(agent=agent, environment=env)

    try:
        if train:
            if debug['general']:
                print('Entering train mode...', file=debug_file, flush=True)
            verbose = 1
            if self_play:
                if debug['general']:
                    print('(train model) Putting weight to model_q for first time...', file=debug_file, flush=True)
                model_q.put_nowait(model.get_weights())
                verbose = 0
            for step in range(episodes//reset_every):
                if backend == 'ddqn':
                    dqn.fit(env, nb_steps=0,
                            nb_episodes=reset_every, visualize=False,
                            verbose=verbose, save_every=50000 if not self_play else 10**9, save_path='{}/duel_dqn'.format(weights_path))  # , debug_callback=debug_callback)
                else:
                    dqn.run(num_episodes=reset_every, episode_finished=episode_finished)

                if self_play:
                    win_ep, lose_ep = 1, 1
                    enemy_win_ep, enemy_lose_ep = 1, 1
                    for msg in msg_q:
                        if second == msg['second']:
                            if msg['msg'] == 'win':
                                win_ep += 1
                            elif msg['msg'] == 'lose':
                                lose_ep += 1
                            #elif debug['general']:
                            #    print('Unknown msg type', msg['msg'])
                        else:
                            if msg['msg'] == 'win':
                                enemy_win_ep += 1
                            elif msg['msg'] == 'lose':
                                enemy_lose_ep += 1
                            #elif debug['general']:
                            #    print('Unknown msg type', msg['msg'])
                    if win_ep/lose_ep > 1.3:
                        if debug['general']:
                            print('(train model, second: {}) Putting weight to model_q because of win, winrate: {}'.format(second, win_ep/lose_ep), file=debug_file, flush=True)
                        model_q.put_nowait(model.get_weights())
                        dqn.save_weights('{}/duel_dqn_sp_{}.h5f'.format(weights_path, step))

                    elif enemy_win_ep/enemy_lose_ep > 1.3:
                        if debug['general']:
                            print('(train model, second: {}) Getting weight from model_q because of lose, winrate: {}'.format(second, win_ep/lose_ep), file=debug_file, flush=True)
                        while model_q.empty():
                            pass
                        model.set_weights(model_q.get_nowait())
                        if debug['general']:
                            print('Success')

                    elif debug['general']:
                        print('No one win or lose, winrate:', win_ep/lose_ep, enemy_win_ep/enemy_lose_ep, file=debug_file, flush=True)
                    if second:
                        if debug['general']:
                            print('Second done!', file=debug_file, flush=True)
                        msg_q.append({'second': second, 'msg': 'done'})
                    else:
                        if debug['general']:
                            print('Waiting when second will be done', file=debug_file, flush=True)
                        while msg_q[-1]['msg'] != 'done':
                            pass

                        # clear listproxy: https://stackoverflow.com/questions/23499507/how-to-clear-the-content-from-a-listproxy/45829822
                        msg_q[:] = []
                        if debug['general']:
                            print('msg_q clear', file=debug_file, flush=True)
                else:
                    msg_q[:] = []
                #if self_play:
                #    if debug['general']:
                #        print('(train model) Putting weight to model_q...', file=debug_file, flush=True)
                #    model_q.put_nowait(model.get_weights())
        else:
            if self_play:
                if debug['general']:
                    print('(freezed_model) Resetting weights for first time...', file=debug_file, flush=True)
                while model_q.empty():
                    pass
                model.set_weights(model_q.get_nowait())
                if debug['general']:
                    print('(freezed_model) Success!', file=debug_file, flush=True)
            for _ in range(episodes // reset_every):
                if backend == 'ddqn':
                    dqn.test(env, nb_episodes=reset_every, visualize=False)
                else:
                    dqn.run(num_episodes=reset_every, episode_finished=episode_finished, testing=True)
                if self_play:
                    if debug['general']:
                        print('(freezed_model) Resetting weights...', file=debug_file, flush=True)
                    while model_q.empty():
                        pass
                    model.set_weights(model_q.get_nowait())
                    if debug['general']:
                        print('(freezed_model) Success!', file=debug_file, flush=True)
    except:
        print('Exception:', file=env.debug_file, flush=True)
        traceback.print_exc()

    # After training is done, we save the final weights.
    #dqn.save_weights('duel_dqn_{}_weights.h5f'.format('test'), overwrite=True)
    #print('Weights saved', file=env.debug_file, flush=True)

    # Finally, evaluate our algorithm for 5 episodes.
    #dqn.test(env, nb_episodes=5, visualize=False)


class RLClient(Client):
    """
    maps = ['PillMap', 'PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
    cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']
    """
    def __init__(self, args, second, manager, debug_file=None, train=False, model_q=None, msg_q=None, self_play=False):
        """print('RL client init start')
        self.msg_in = Queue()
        self.msg_out = Queue()
        self.model = None
        self.debug = debug_file is not None
        self.env = RLEnv(self.msg_in, self.msg_out, debug_file=debug_file)
        self.thread = Thread(name=name, target=self.run_strategy, daemon=True)
        self.thread.start()"""
        self.manager = manager
        self.msg_in = self.manager.Queue()
        self.msg_out = self.manager.Queue()
        print('Second:', second, 'model_q:', model_q)
        self.model_q = self.manager.Queue() if model_q is None else model_q
        self.msg_q = self.manager.list() if msg_q is None else msg_q
        self.debug = {'general': True, 'obs': False, 'msg': False}
        self.resume_weights = None if args.resume == -1 else args.resume
        self.process = mp.Process(target=run_strategy, args=(self.msg_in, self.msg_out, self.model_q, self.msg_q, debug_file,
                                                             self.debug, self.resume_weights, train, args.n, args.reset_every,
                                                             self_play, second, 'weights/' + args.env_map + '_' + args.env_car,
                                                             args.dueling1 if not second else args.dueling2), daemon=True)
        self.process.start()
        print('RL client started')

    def send_message(self, t, d):
        msg = {
            'type': t,
            'params': d
        }
        self.msg_in.put_nowait(json.dumps(msg))
        #print('send_msg:', msg)

    @asyncio.coroutine
    def get_command(self):
        #start = default_timer()
        while self.msg_out.empty():
            pass
        #end = default_timer()
        #print('get_cmd time:', end-start)
        try:
            msg = self.msg_out.get_nowait()
            #print('get_command:', msg)
            return json.loads(msg)
        except:
            traceback.print_exc()
            raise Exception('get_command exc')

