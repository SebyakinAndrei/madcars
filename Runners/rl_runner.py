import asyncio

from mechanic.strategy import FileClient
from mechanic.game import Game
from mechanic.constants import MATCHES_COUNT
import argparse


class GameServer:
    def __init__(self):
        self.clients = []

    @asyncio.coroutine
    def connection_handler(self, client_reader, client_writer):
        client = TcpClient(client_reader, client_writer)
        is_success = yield from client.set_solution_id()

        clients_count = len(self.clients)

        if clients_count < 2:
            if is_success:
                clients_count += 1
                print('{} clients connected'.format(clients_count))
                self.clients.append(client)
            else:
                loop.stop()
        else:
            client_writer.close()

        game_future = None
        if clients_count == 2:
            game = Game(self.clients, Game.generate_matches(MATCHES_COUNT))
            print('game started')
            game_future = asyncio.ensure_future(game.game_loop())

        if game_future:
            done, pending = yield from asyncio.wait([game_future])
            if not pending:
                loop.stop()
            print('game done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RLRunner for MadCars')

    parser.add_argument('-f', '--fp', type=str, nargs='?',
                        help='Path to executable with strategy for first player', default='keyboard')
    parser.add_argument('--fpl', type=str, nargs='?', help='Path to log for first player')

    parser.add_argument('-s', '--sp', type=str, nargs='?',
                        help='Path to executable with strategy for second player', default='keyboard')
    parser.add_argument('--spl', type=str, nargs='?', help='Path to log for second player')
    parser.add_argument('--nodraw', action='store_true')
gs = GameServer()

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.start_server(gs.connection_handler, '0.0.0.0', 8000))
try:
    loop.run_forever()
finally:
    loop.close()
