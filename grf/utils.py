import argparse
import asyncio
import time
import os
import subprocess
import signal

from .grf import BaseNewGRF

OPENTTD = '/home/dp/Projects/OpenTTD/build-release/openttd'


import asyncio
import struct

from pprint import pprint

SERVER = 'localhost'
PORT = 3977
PASSWORD = b'123'
ADMIN_NAME = 'grf-py live reload'
ADMIN_VERSION = '1.0'

ADMIN_JOIN, ADMIN_QUIT, ADMIN_UPDATE_FREQUENCY, ADMIN_POLL, ADMIN_CHAT, ADMIN_RCON, ADMIN_GAMESCRIPT, ADMIN_PING, ADMIN_EXTERNAL_CHAT = range(9)
ACTION_JOIN, ACTION_LEAVE, ACTION_SERVER_MESSAGE, ACTION_CHAT, ACTION_CHAT_COMPANY, ACTION_CHAT_CLIENT, ACTION_GIVE_MONEY, ACTION_NAME_CHANGE, ACTION_COMPANY_SPECTATOR, ACTION_COMPANY_JOIN, ACTION_COMPANY_NEW, ACTION_KICKED = range(12)
DESTTYPE_BROADCAST, DESTTYPE_TEAM, DESTTYPE_CLIENT = range(3)


def ttd_pack(fmt, *data):
    fl, dl = '<', []
    for c, d in zip(fmt, data):
        if isinstance(d, str):
            d = d.encode('utf8')
        dl.append(d)
        if c != 'z':
            fl += c
        else:
            fl += f'{len(d)}sB'
            dl.append(0)

    return struct.pack(fl, *dl)


def make_packet(packet_type, fmt=None, *args):
    data = ttd_pack(fmt, *args) if fmt else b''
    size = len(data) + 3
    return bytes((size & 255, size >> 8, packet_type)) + data


async def reload_newgrfs():
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(SERVER, PORT),
        timeout=3)

    writer.write(make_packet(ADMIN_JOIN, 'zzz', PASSWORD, ADMIN_NAME, ADMIN_VERSION))
    writer.write(make_packet(ADMIN_RCON, 'z', 'reload_newgrfs'))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


def build_func(g, grf_file, args):
    print(f'Building {grf_file}')
    g.write(grf_file, clean_build=False if args is None else args.clean)


def watch_func(g, grf_file, args):
    import watchdog.events
    import watchdog.observers

    class FileChangeHandler(watchdog.events.FileSystemEventHandler):
        def __init__(self, file_list):
            self.file_list = file_list
            self.modified = set()

        def reset(self):
            self.modified = set()

        def on_modified(self, event):
            if event.src_path in self.file_list:
                self.modified.add(event.src_path)

    print(f'Building {grf_file}')
    watched_files = g.write(grf_file)

    # print(f'Starting OpenTTD.')
    # process = subprocess.Popen([OPENTTD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, shell=False)
    asyncio.run(reload_newgrfs())

    print(f'Watching {len(watched_files)} files for changes...')

    event_handler = FileChangeHandler(watched_files)
    observer = watchdog.observers.Observer()

    for file_path in watched_files:
        observer.schedule(event_handler, os.path.dirname(file_path), recursive=False)

    observer.start()

    try:
        while True:
            time.sleep(1)
            if event_handler.modified:
                time.sleep(1)
                modified_files = ', '.join(event_handler.modified)
                print(f"{modified_files} has been modified, rebuilding grf")
                event_handler.reset()
                g.write(grf_file)
                print(f'Reloading newgrfs.')
                asyncio.run(reload_newgrfs())
                # process.stdin.write('reload_newgrfs\n')
                # output = process.stdout.read()
                # print('OUT', output)
    except KeyboardInterrupt:
        observer.stop()

    # process.send_signal(signal.SIGKILL)
    observer.join()


def init_id_map_func(g, grf_file, args):
    with open('id_map.json', 'w') as f:
        f.write('{"version": 1, "index": {}}')
    print(f'Created id_map.json')


def main(g, grf_file):
    assert isinstance(g, BaseNewGRF)
    assert isinstance(grf_file, str)
    parser = argparse.ArgumentParser(description=g.name)

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(title='commands', dest='command')

    # Create a parser for the 'build' command
    build_parser = subparsers.add_parser('build', help='Build newgrf')
    build_parser.add_argument('--clean', action='store_true', help='Clean build (don''t use sprite cache)')
    # create_parser.add_argument('--size', type=int, required=True, help='Size of the item')
    build_parser.set_defaults(func=build_func)

    # Create a parser for the 'build' command
    watch_parser = subparsers.add_parser('watch', help='Build newgrf and rebuild if any files changed')
    watch_parser.set_defaults(func=watch_func)

    watch_parser = subparsers.add_parser('init_id_map', help='Initialize the automatic id index (id_map.json)')
    watch_parser.set_defaults(func=init_id_map_func)

    args = parser.parse_args()

    if args.command is None:
        build_func(g, grf_file, None)
    else:
        args.func(g, grf_file, args)

