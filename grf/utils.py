import argparse
import asyncio
import time
import os
import subprocess
import signal
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

from .grf import BaseNewGRF

OPENTTD = '/home/dp/Projects/OpenTTD/build-release/openttd'


import asyncio
import struct

from pprint import pprint

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


async def admin_logging(queue, admin_addr):
    if admin_addr is None:
        return

    server, port, password = admin_addr

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(server, port),
        timeout=3)

    writer.write(make_packet(ADMIN_JOIN, 'zzz', password, ADMIN_NAME, ADMIN_VERSION))

    prev_msg = ''
    while True:
        msg = await queue.get()
        if msg is None:
            queue.task_done()
            break
        if msg.endswith('\n'):
            writer.write(make_packet(ADMIN_EXTERNAL_CHAT, 'zHzz', 'grf-py', 1, '', prev_msg + msg.rstrip('\n')))
            prev_msg = ''
        else:
            prev_msg += msg
        queue.task_done()

    writer.write(make_packet(ADMIN_RCON, 'z', 'reload_newgrfs'))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


def build_func(g, grf_file, args):
    print(f'Building {grf_file}')
    g.write(
        grf_file,
        clean_build=False if args is None else args.clean,
        debug_zoom_levels=False if args is None else args.debug_zoom_levels,
    )


async def async_compile(g, grf_file, queue, admin_addr):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    def compile_func(g, grf_file):
        watched = g.write(grf_file)
        queue.put_nowait(None)
        g._context.print(f'Reloading newgrfs.')
        return watched

    compilation = loop.run_in_executor(executor, compile_func, g, grf_file)
    logging = admin_logging(queue, admin_addr)
    group = asyncio.gather(compilation, logging)

    try:
        res = await group
        return res[0]
    except BaseException as e:
        group.cancel()
        tb = traceback.format_exc()
        for l in tb.split('\n'):
            g._context.print(l)
        return None


def watch_func(g, grf_file, args):
    import watchdog.events
    import watchdog.observers

    admin_addr = None
    if args.live_reload is not None:
        s = args.live_reload
        port = 3977
        addr = 'localhost'
        if ':' in s:
            s, port_str = s.rsplit(':', 1)
            port = int(port_str)
        if '@' in s:
            s, addr = s.split('@', 1)
        pw = s
        print(f'Admin port connection string: {pw}@{addr}:{port}')
        admin_addr = (addr, port, pw)
    else:
        print(f'No admin port specified, live reload unavailable')

    class FileChangeHandler(watchdog.events.FileSystemEventHandler):
        def __init__(self, file_list):
            self.file_list = file_list
            self.modified = set()

        def reset(self):
            self.modified = set()

        def on_modified(self, event):
            if self.file_list is None:
                return
            if Path(event.src_path).resolve() in self.file_list:
                self.modified.add(event.src_path)

    queue = None

    def print_func(msg, end='\n', flush=False):
        nonlocal queue
        if queue is None:
            return
        queue.put_nowait(msg + end)

    g._context.add_print_handler(print_func)

    async def run_compile():
        event_handler = FileChangeHandler(None)
        observer = watchdog.observers.Observer()
        observer.start()
        nonlocal queue
        try:
            while True:
                time.sleep(1)
                if event_handler.modified or event_handler.file_list is None:
                    time.sleep(1)
                    modified_files = ', '.join(event_handler.modified)
                    event_handler.reset()
                    queue = asyncio.Queue()
                    if not modified_files:
                        g._context.print(f"Building {grf_file}")
                    else:
                        g._context.print(f"{modified_files} has been modified, rebuilding {grf_file}")
                    prev_watched = event_handler.file_list
                    watched_files = await async_compile(g, grf_file, queue, admin_addr)
                    if watched_files is None:
                        if prev_watched is None:
                            g._context.print(f'Grf build failed, retrying in 10 seconds...')
                            time.sleep(10)
                    else:
                        event_handler.file_list = {Path(f).resolve() for f in watched_files}
                        if prev_watched != watched_files:
                            g._context.print(f'Watching {len(watched_files)} files for changes...')
                        for file_path in watched_files:
                            if prev_watched is not None and file_path in prev_watched:
                                continue
                            observer.schedule(event_handler, os.path.dirname(file_path), recursive=False)

        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    asyncio.run(run_compile())
    # process.send_signal(signal.SIGKILL)


def init_id_map_func(g, grf_file, args):
    with open('id_map.json', 'w') as f:
        f.write('{"version": 1, "index": {}}')
    print(f'Created id_map.json')


def main(g, grf_file, commands=None):
    assert isinstance(g, BaseNewGRF)
    assert isinstance(grf_file, str)
    parser = argparse.ArgumentParser(description=getattr(g, 'name', None))

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(title='commands', dest='command')

    # Create a parser for the 'build' command
    build_parser = subparsers.add_parser('build', help='Build newgrf')
    build_parser.add_argument('--clean', action='store_true', help='Clean build (don''t use sprite cache)')
    build_parser.add_argument('--debug-zoom-levels', action='store_true', help='Recolor sprites according to their zoom level: 4x - red, 2x - blue, 1x - green, out-2x - cyan, out-4x - yellow, out-8x - magenta')
    # create_parser.add_argument('--size', type=int, required=True, help='Size of the item')
    build_parser.set_defaults(func=build_func)

    # Create a parser for the 'build' command
    watch_parser = subparsers.add_parser('watch', help='Build newgrf and rebuild if any files changed')
    watch_parser.set_defaults(func=watch_func)
    watch_parser.add_argument('--live-reload', type=str, help='Admin port to connect in a form password@address:port')

    watch_parser = subparsers.add_parser('init_id_map', help='Initialize the automatic id index (id_map.json)')
    watch_parser.set_defaults(func=init_id_map_func)

    for c in commands or []:
        cmd_parser = subparsers.add_parser(c['name'], help=c['help'])
        if 'add_args' in c:
            c['add_args'](cmd_parser)
        cmd_parser.set_defaults(func=c['handler'])

    args, unknown = parser.parse_known_args()

    if args.command is None:
        build_func(g, grf_file, None)
    else:
        args.func(g, grf_file, args)

