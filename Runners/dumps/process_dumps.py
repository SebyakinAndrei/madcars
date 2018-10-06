import json
import math
import os

def norm_angle(theta):
    ''' Normalize an angle in radians to [0, 2*pi) '''
    angle = theta % (2*math.pi)
    if angle < 0:
        angle = 2*math.pi - angle
    return angle

prev_pos = (-1, -1)
prev_angle = -1


def get_observation(msg):
    global prev_pos, prev_angle
    observation = []
    for car in ['my_car', 'enemy_car']:
        data = msg[car]
        # print('angle ({}): {}'.format(car, data[1], norm_angle(data[1])))
        if data[2] == -1:
            for i in [0, 3, 4]:
                data[i] = (1200 - data[i][0], data[i][1])
        observation.extend([data[2], norm_angle(data[1]), data[3][0], data[3][1], data[4][0], data[4][1]])
    observation.append(min(observation[3], observation[5]) - msg['params'].get('deadline_position', 0))
    if prev_pos[0] == -1:
        observation.extend([0.0, 0.0])
    else:
        observation.extend([(observation[2] + observation[4]) / 2 - prev_pos[0],
                            (observation[3] + observation[5]) / 2 - prev_pos[1]])
    prev_pos = ((observation[2] + observation[4]) / 2, (observation[3] + observation[5]) / 2)
    if prev_angle == -1:
        observation.append(0.0)
    else:
        observation.append(observation[1] - prev_angle)
    prev_angle = observation[1]

    return observation


def process(directory):
    global prev_pos, prev_angle
    # [0,"stop",[[300.0,300.0],0.0,1,[329.0,295.0,0.0],[422.0,295.0,0.0]]]
    states = []
    actions = []
    terminal = []
    reward = []
    for fname in os.listdir(directory):
        with open(directory + '/' + fname) as f:
            prev_pos = (-1, -1)
            prev_angle = -1
            raw_my, raw_enemy = json.loads(f.readline()), json.loads(f.readline())
            if not raw_my['win']:
                print('Skipping {}...'.format(fname))
                continue
            obs_my, obs_enemy = raw_my['dump'], raw_enemy['dump']
            for step in len(obs_my):
                actions.append(obs_my[step][1])
                states.append(get_observation({'my_car': obs_my[step][2], 'enemy_car': obs_enemy[step][2]}))
                if step < len(obs_my) - 1:
                    terminal.append(False)
                    reward.append(1)
                else:
                    terminal.append(True)
                    reward.append(10000)
    return states, actions, terminal, reward
