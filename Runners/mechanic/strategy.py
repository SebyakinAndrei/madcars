import datetime
import gzip
import json

import os
import random

import pyglet
import asyncio
from subprocess import Popen, PIPE

from mechanic.constants import MAX_EXECUTION_TIME, REQUEST_MAX_TIME
import traceback


class Client(object):
    def get_command(self):
        pass

    def close(self):
        pass

    def send_message(self, t, d):
        pass

    def save_log_to_disk(self, log, path):
        pass

    def get_solution_id(self):
        return random.randint(11000, 12000)


class KeyboardClient(Client):
    @property
    def KEY_COMMAND_MAP(self):
        return {
            pyglet.window.key.MOTION_LEFT: 'left',
            pyglet.window.key.MOTION_RIGHT: 'right',
            pyglet.window.key.MOTION_DOWN: 'stop',
        } if not self.second else {
            pyglet.window.key.A: 'left',
            pyglet.window.key.D: 'right',
            pyglet.window.key.S: 'stop'
        }

    def __init__(self, window, second=False, first_client=None):
        self.second = second
        self.last_pressed_button = pyglet.window.key.MOTION_DOWN if not self.second else pyglet.window.key.S
        self.first_client = first_client

        @window.event
        def on_key_press(symbol, _):
            if symbol in self.KEY_COMMAND_MAP:
                self.last_pressed_button = symbol
            elif self.second and (symbol in [pyglet.window.key.MOTION_RIGHT, pyglet.window.key.MOTION_LEFT, pyglet.window.key.MOTION_DOWN]):
                self.first_client.last_pressed_button = symbol

        @window.event
        def on_key_release(symbol, _):
            if symbol in [pyglet.window.key.MOTION_RIGHT, pyglet.window.key.MOTION_LEFT]:
                if not self.second:
                    self.last_pressed_button = pyglet.window.key.MOTION_DOWN
                else:
                    self.first_client.last_pressed_button = pyglet.window.key.MOTION_DOWN
            elif self.second and (symbol in [pyglet.window.key.A, pyglet.window.key.D]):
                self.last_pressed_button = pyglet.window.key.S


    @asyncio.coroutine
    def get_command(self):
        return {'command': self.KEY_COMMAND_MAP.get(self.last_pressed_button, 'stop')}

    def save_log_to_disk(self, log, path):
        pass


class FileClient(Client):
    def __init__(self, path_to_script, path_to_log=None):
        self.process = Popen(path_to_script, stdout=PIPE, stdin=PIPE)
        self.last_message = None
        if path_to_log is None:
            base_dir = os.getcwd()
            now = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M-%S.log.gz')
            self.path_to_log = os.path.join(base_dir, now)
        else:
            self.path_to_log = path_to_log

    def send_message(self, t, d):
        try:
            msg = {
                'type': t,
                'params': d
            }
            msg_bytes = '{}\n'.format(json.dumps(msg)).encode()

            self.process.stdin.write(msg_bytes)
            self.process.stdin.flush()
            #print('msg:', msg_bytes)
        except:
            traceback.print_exc()

    @asyncio.coroutine
    def get_command(self):
        line = None
        try:
            line = self.process.stdout.readline().decode('utf-8')
            #print('line: ('+line+')')
            state = json.loads(line)
            return state
        except Exception as e:
            traceback.print_exc()
            print('line: (' + line + ')')
            return {'command': 'stop', 'debug': str(e)}

    def save_log_to_disk(self, log, _):
        try:
            with gzip.open(self.path_to_log, 'w') as f:
                f.write(json.dumps(log).encode())

            return {
                'filename': os.path.basename(self.path_to_log),
                'is_private': True,
                'location': self.path_to_log
            }
        except:
            traceback.print_exc()


class TcpClient(Client):
    EXECUTION_LIMIT = datetime.timedelta(seconds=MAX_EXECUTION_TIME)

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.execution_time = datetime.timedelta()
        self.solution_id = None

    def save_log_to_disk(self, log, path):
        location = path.format(str(self.solution_id) + '.gz')

        with gzip.open(location, 'wb') as f:
            f.write(json.dumps(log).encode())

        return {
            'filename': os.path.basename(location),
            'is_private': True,
            'location': location
        }

    @asyncio.coroutine
    def set_solution_id(self):
        hello_json = yield from asyncio.wait_for(self.reader.readline(), timeout=REQUEST_MAX_TIME)
        try:
            self.solution_id = json.loads(hello_json.decode('utf-8')).get('solution_id')
        except ValueError:
            pass

        return bool(self.solution_id)

    def send_message(self, t, d):
        msg = {
            'type': t,
            'params': d
        }
        msg_bytes = '{}\n'.format(json.dumps(msg)).encode()
        self.writer.write(msg_bytes)

    @asyncio.coroutine
    def get_command(self):
        try:
            before = datetime.datetime.now()
            z = yield from asyncio.wait_for(self.reader.readline(), timeout=REQUEST_MAX_TIME)
            if not z:
                raise ConnectionError('Connection closed')
            self.execution_time += (datetime.datetime.now() - before)
            if self.execution_time > self.EXECUTION_LIMIT:
                raise Exception('sum timeout error')
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError('read timeout error')
        try:
            z = json.loads(z.decode())
        except ValueError:
            z = {'debug': 'cant pars json'}

        return z

    def close(self):
        self.writer.close()

    def get_solution_id(self):
        return self.solution_id
