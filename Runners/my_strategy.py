import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
stdout = sys.stdout
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdout = open(os.devnull, 'w')

import json
import random
import math
import traceback
from queue import Queue



def norm_angle(theta):
    ''' Normalize an angle in radians to [0, 2*pi) '''
    angle = theta % (2*math.pi)
    if angle < 0:
        angle = 2*math.pi - angle
    return angle


class ActionSpace:
    def sample(self):
        return random.choice([0, 1, 2])


class RLEnv:
    def __init__(self, q_in=None, q_out=None, msg_q=None, second=False, debug_file=None, debug=None, inference=False, new_conf=None):
        if inference:
            self.output = self.output_std
            self.input = self.input_std

        self.new_conf = new_conf

        self.q_in = q_in
        self.q_out = q_out
        self.msg_q = msg_q
        self.second = second
        self.last_obs = []
        self.debug_file = debug_file
        #print('debug_file:', debug_file, type(debug_file))
        #if isinstance(debug_file, str):
        #    self.debug_file = open(debug_file, 'w')
        if debug is None:
            self.debug = {'general': False, 'obs': False, 'msg': False}
        else:
            self.debug = debug
        self.first_match = True
        self.my_lives = -1
        self.env_car = -1
        self.env_map = -1
        self.need_end = (False,)
        self.init_obs = None
        self.actions = ['left', 'right', 'stop']
        self.reversed = None
        self.prev_pos = (-1, -1)
        self.prev_angle = -1
        self.ticks_before_die = -1

        self.action_space = ActionSpace()
        if self.debug['general']:
            print('RL ENV STARTED', file=self.debug_file, flush=True)

    def input(self):
        while self.q_in.empty():
            pass
        return self.q_in.get_nowait()

    def input_std(self):
        return input()

    def output_std(self, out, debug=None):
        if debug is not None:
            out['debug'] = debug
        print(json.dumps(out), flush=True)

    def output(self, out, debug=None):
        self.q_out.put_nowait(json.dumps(out))

    def get_observation(self, msg):
        try:
            observation = []
            if msg['type'] == 'tick':
                for car in ['my_car', 'enemy_car']:
                    data = msg['params'][car]
                    #print('angle ({}): {}'.format(car, data[1], norm_angle(data[1])))
                    if data[2] == -1:
                        if self.reversed is None:
                            self.reversed = True
                        for i in [0, 3, 4]:
                            data[i] = (1200 - data[i][0], data[i][1])
                    else:
                        if self.reversed is None:
                            self.reversed = False
                    observation.extend([data[2], norm_angle(data[1]), data[3][0], data[3][1], data[4][0], data[4][1]])
                observation.append(min(observation[3], observation[5]) - msg['params'].get('deadline_position', 0))
                if self.prev_pos[0] == -1:
                    observation.extend([0.0, 0.0])
                else:
                    observation.extend([(observation[2] + observation[4])/2 - self.prev_pos[0],
                                        (observation[3] + observation[5])/2 - self.prev_pos[1]])
                self.prev_pos = ((observation[2] + observation[4])/2, (observation[3] + observation[5])/2)
                if self.prev_angle == -1:
                    observation.append(0.0)
                else:
                    observation.append(observation[1] - self.prev_angle)
                self.prev_angle = observation[1]
                if self.debug['obs']:
                    print('({}):'.format(len(observation)), observation, file=self.debug_file, flush=True)
            return observation
        except:
            if self.debug['general']:
                print('GET OBS Exception', file=self.debug_file, flush=True)
            traceback.print_exc()

    def reset(self, msg=None, get_init_obs=True, caller=None):
        try:
            self.need_end = (False,)
            if self.debug['general']:
                print('Reset called! caller: {}'.format(caller), msg, file=self.debug_file, flush=True)

            if msg is None:
                msg = json.loads(self.input())
                if self.debug['msg']:
                    print('msg (reset):', msg, file=self.debug_file, flush=True)
                if (msg['type'] == 'tick') and get_init_obs:
                    if self.debug['general']:
                        print('Reset done, exitpoint 1', file=self.debug_file, flush=True)
                    obs = self.get_observation(msg)

                    if (obs is None) and self.debug['general']:
                        print('OBS is None', file=self.debug_file, flush=True)
                    if self.debug['obs']:
                        print('obs:', obs, file=self.debug_file, flush=True)

                    #self.q_out.put_nowait(json.dumps({'command': 'stop'}))
                    self.output({'command': 'stop', 'debug': 'endpoint1'})
                    return obs

            if msg['type'] == 'new_match':
                if isinstance(self.new_conf, list):
                    self.new_conf.append(msg)
                self.prev_pos = (-1, -1)
                self.reversed = None
                self.env_car = msg['params']['proto_car']['external_id']
                self.env_map = msg['params']['proto_map']['external_id']
                #self.button_y = msg['params']['proto_car']['button_poly'][0][1]
                ##print('Button:', msg['params']['proto_car']['button_poly'][0], file=self.debug_file, flush=True)
                if self.first_match:
                    self.first_match = False
                else:
                    if msg['params']['my_lives'] < self.my_lives:
                        self.need_end = (True, -10000)
                        if self.msg_q is not None:
                            self.msg_q.append({'second': self.second, 'msg': 'lose'})
                    else:
                        self.need_end = (True, 10000 - self.ticks_before_die)
                        if self.msg_q is not None:
                            self.msg_q.append({'second': self.second, 'msg': 'win'})
                    if self.debug['general']:
                        print('Reward:', self.need_end[1], file=self.debug_file, flush=True)
                self.ticks_before_die = 0
                if get_init_obs:
                    msg = json.loads(self.input())
                    if self.debug['general']:
                        print('Reset done, exitpoint 2', file=self.debug_file, flush=True)
                    #self.q_out.put_nowait(json.dumps({'command': 'stop'}))
                    self.output({'command': 'stop', 'debug': 'endpoint2'})
                    return self.get_observation(msg)
                self.my_lives = msg['params']['my_lives']

            if self.debug['general']:
                print('Reset done, exitpoint 3', file=self.debug_file, flush=True)
        except:
            if self.debug['general']:
                print('GET RESET Exception', file=self.debug_file, flush=True)
            traceback.print_exc()

    def execute(self, action, debug=None):
        observation, reward, done, info = self.step(action, debug=debug)
        return observation, done, reward

    def step(self, action, debug=None):
        try:
            #print('step {}'.format(action), file=self.debug_file, flush=True)
            if 0 <= action <= 2:
                if not self.reversed:
                    action = self.actions[action]
                elif action == 1:
                    action = self.actions[0]
                elif action == 0:
                    action = self.actions[1]
                elif action == 2:
                    action = self.actions[2]
            else:
                action = self.actions[2]
            #print('before get msg', file=self.debug_file, flush=True)
            msg = self.input()
            if self.debug['msg']:
                print('msg (step):', msg, file=self.debug_file, flush=True)
            msg = json.loads(msg)
            observation, reward, done, info = [], 1.0, False, {}
            if msg['type'] == 'new_match':
                if isinstance(self.new_conf, list):
                    self.new_conf.append(msg)
                self.reset(msg=msg, get_init_obs=False, caller='step')
                if self.need_end[0]:
                    reward = self.need_end[1]
                    done = True
                    observation = self.last_obs
                    self.need_end = (False,)
            else:
                observation = self.get_observation(msg)
                self.last_obs = observation
                #if debug is None:
                #    debug = ''
                #self.q_out.put_nowait(json.dumps({'command': action, 'debug': debug}))
                self.output({'command': action, 'debug': 'action: {}'.format(action)})

            #if action == 'stop':
            #    reward -= 1
            #reward -= 1
            self.ticks_before_die += 1
            #reward += abs(round(observation[-3]))
            if done and self.debug['general']:
                print('Done!', file=self.debug_file, flush=True)
            #print(observation, reward, done, info, file=self.debug_file, flush=True)
            return observation, reward, done, info
        except:
            if self.debug['general']:
                print("EXCEPTION (step)", file=self.debug_file, flush=True)
            traceback.print_exc()


    def close(self):
        if self.debug['general']:
            print('Closed.', file=self.debug_file, flush=True)
        self.debug_file.close()


def smartguy(new_conf, debug_file=None):
    tick = 0
    while True:
        msg = json.loads(input())
        if msg['type'] == 'new_match':
            new_conf.append(msg)
            return
        cmd = {'command': 'stop', 'debug': 'smartguy!'}
        if tick < 20:
            cmd = {'command': 'stop', 'debug': 'smartguy! tick < 20'}
        else:
            #my_pos = msg['params']['my_car'][0]
            #enemy_pos = msg['params']['enemy_car'][0]
            my_angle = norm_angle(msg['params']['my_car'][1])
            while my_angle > math.pi:
                my_angle -= 2.0 * math.pi

            while my_angle < -math.pi:
                my_angle += 2.0 * math.pi

            #print('angle:', my_angle, file=debug_file, flush=True)

            if my_angle > math.pi/4.5:#0.25:
                cmd = {'command': 'left', 'debug': 'smartguy!'}
            elif my_angle < math.pi/4.5:#-0.25:
                cmd = {'command': 'right', 'debug': 'smartguy!'}
        print(json.dumps(cmd), flush=True)
        tick += 1


def run_strategy():
    import numpy as np

    debug = {'general': False, 'obs': False, 'msg': False}
    backend = 'ddqn'
    weights_path = 'weights'
    #debug_file = open(weights_path + '/' + debug_file, 'w') if debug_file is not None else None
    # Get the environment and extract the number of actions.
    #env = self.env
    new_conf = []
    debug_file = None#open('strat_debug.txt', 'w')
    env = RLEnv(debug=debug, debug_file=debug_file, inference=True, new_conf=new_conf)
    np.random.seed(123)
    nb_actions = 3

    model, dqn = None, None
    if backend == 'ddqn':
        from keras.models import Sequential
        from keras.layers import Dense, Activation, Flatten
        from keras.optimizers import Adam

        from rl.agents.dqn import DQNAgent
        from rl.policy import BoltzmannQPolicy
        from rl.memory import SequentialMemory

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
                       enable_dueling_network=False, dueling_type='avg', target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        #if load_weights is not None:
        #    path = '{}/duel_dqn_{}.h5f'.format(weights_path, load_weights)
        #    dqn.load_weights(path)
        #    if debug['general']:
        #        print('Weights loaded! {}'.format(path), file=env.debug_file, flush=True)
    else:
        from tensorforce.agents import DQFDAgent
        from tensorforce.execution import Runner

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

    cars = {1: 'Buggy', 2: 'Bus', 3: 'SquareWheelsBuggy'}
    maps = {1: 'PillMap', 2: 'PillHubbleMap', 3: 'PillHillMap', 4: 'PillCarcassMap', 5: 'IslandMap', 6: 'IslandHoleMap'}
    done_matches = {'IslandMap_Buggy', 'IslandMap_Bus', 'PillCarcassMap_Buggy', 'PillHubbleMap_Buggy'} #'PillHubbleMap_Bus',
            #'PillMap_Buggy'}
    global stdout, stderr
    sys.stdout = stdout
    sys.stderr = stderr
    if backend == 'ddqn':
        first_match = True
        while True:
            try:
                smart = False
                if first_match:
                    first_match = False
                    env.reset()
                    #env.first_match = True
                msg = new_conf[-1]
                map_car = maps[msg['params']['proto_map']['external_id']] + '_' + cars[msg['params']['proto_car']['external_id']]
                if map_car not in done_matches:
                    smart = True
                else:
                    dqn.load_weights(weights_path + '/' + map_car + '/duel_dqn_101.h5f')
                if smart:
                    smartguy(new_conf, debug_file=debug_file)
                else:
                    dqn.test(env, nb_episodes=1, visualize=False, verbose=0)
            except Exception as e:
                print(json.dumps({'command': 'stop', 'debug': 'after except'.format(str(e))}), flush=True)
                smartguy(new_conf, debug_file=debug_file)
            # / Updating weights comes here... /
    else:
        dqn.run(num_episodes=1, testing=True)


if __name__ == '__main__':
    run_strategy()


"""
python localrunner.py -f "agent" -s "agent" -rl_env -env_car Buggy -env_map IslandMap -n 5000 -reset_every 10
python localrunner.py -f "agent" -s "../smartguy/main.exe" -rl_env -env_car SquareWheelsBuggy -env_map IslandMap -n 500000 -train1 --nodraw
"""