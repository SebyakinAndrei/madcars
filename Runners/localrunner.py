from itertools import product

import pyglet
import pymunk.pyglet_util
import argparse

import asyncio
import traceback
from asyncio import events
from multiprocessing import Manager
from os.path import isdir
from os import mkdir

from mechanic.game import Game
from mechanic.strategy import KeyboardClient, FileClient
from rl_env import RLClient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LocalRunner for MadCars')

    parser.add_argument('-f', '--fp', type=str, nargs='?',
                        help='Path to executable with strategy for first player', default='keyboard')
    parser.add_argument('--fpl', type=str, nargs='?', help='Path to log for first player')

    parser.add_argument('-s', '--sp', type=str, nargs='?',
                        help='Path to executable with strategy for second player', default='keyboard')
    parser.add_argument('--spl', type=str, nargs='?', help='Path to log for second player')
    parser.add_argument('--nodraw', action='store_true')


    maps = ['PillMap', 'PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
    cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']
    games = [','.join(t) for t in product(maps, cars)]


    parser.add_argument('-m', '--matches', nargs='+', help='List of pairs(map, car) for games', default=games)

    parser.add_argument('-rl_env', action='store_true')
    parser.add_argument('-n', type=int)
    parser.add_argument('-resume', type=int, default=-1)
    parser.add_argument('-reset_every', type=int, default=-1)
    parser.add_argument('-env_car', type=str)
    parser.add_argument('-env_map', type=str)
    parser.add_argument('-train1', action='store_true')
    parser.add_argument('-train2', action='store_true')
    parser.add_argument('-dump', action='store_true')
    parser.add_argument('-dueling1', action='store_true')
    parser.add_argument('-dueling2', action='store_true')

    args = parser.parse_args()

    if args.rl_env:
        args.matches = [args.env_map+','+args.env_car]*(args.n)
        #print('Matches:', args.matches)

    if not args.nodraw:
        window = pyglet.window.Window(1200, 800, vsync=False)
        draw_options = pymunk.pyglet_util.DrawOptions()
        _ = pyglet.clock.ClockDisplay(interval=0.016)

    first_player = args.fp
    second_player = args.sp

    manager = Manager()
    self_play = args.reset_every != -1
    print('Self-play:', self_play)

    if args.rl_env:
        path = 'weights/{}_{}'.format(args.env_map, args.env_car)
        if not isdir(path):
            mkdir(path)
            print('path {} created!'.format(path))

    fc = None
    if args.fp == 'agent':
        fc = RLClient(args, False, manager, debug_file='debug_first.txt', train=args.train1, self_play=self_play)
    elif args.fp == 'keyboard':
        fc = KeyboardClient(window)
    else:
        fc = FileClient(args.fp.split(), args.fpl)

    if args.sp == 'agent':
        sc = RLClient(args, True, manager, debug_file='debug_second.txt',
                      model_q=fc.model_q if args.fp == 'agent' else None,
                      msg_q=fc.msg_q if args.fp == 'agent' else None, train=args.train2, self_play=self_play)
    elif args.sp == 'keyboard':
        sc = KeyboardClient(window, second=True, first_client=fc)
    else:
        sc = FileClient(args.sp.split(), args.spl)

    game = Game([fc, sc], args.matches, extended_save=False, should_dump=args.dump, env_car=args.env_car, env_map=args.env_map)

    @asyncio.coroutine
    def run_game():
        game_future = None
        game_future = asyncio.ensure_future(game.game_loop(nolimit=True))

        if game_future:
            done, pending = yield from asyncio.gather(game_future)
            if not pending:
                loop.stop()
            print('game done')

    try:
        if not args.nodraw:
            loop = events.new_event_loop()
            events.set_event_loop(loop)

            @window.event
            def on_draw():
                if not args.nodraw:
                    pyglet.gl.glClearColor(255,255,255,255)
                    window.clear()
                    game.draw(draw_options)
                game.tick()
                if not game.game_complete:
                    future_message = loop.run_until_complete(game.tick())
                else:
                    winner = game.get_winner()
                    if not args.nodraw:
                        if winner:
                            pyglet.text.Label("Player {} win".format(winner.id), font_name='Times New Roman',
                                          font_size=36,
                                          color=(255, 0, 0, 255),
                                          x=600, y=500,
                                          anchor_x='center', anchor_y='center').draw()
                        else:
                            pyglet.text.Label("Draw", font_name='Times New Roman',
                                              font_size=36,
                                              color=(255, 0, 0, 255),
                                              x=600, y=500,
                                              anchor_x='center', anchor_y='center').draw()


            pyglet.app.run()
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_game())
            try:
                loop.run_forever()
            finally:
                loop.close()
    except:
        traceback.print_exc()


