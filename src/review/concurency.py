# -*- coding: utf-8 -*-

""" python cookbook3 chapter12
"""

import time


import gzip
import io
import glob

from concurrent import futures
from queue import Queue
from threading import Thread, Event


def find_robots(filename):
    robots = set()
    with gzip.open(filename) as f:
        for line in io.TextIOWrapper(f, encoding="ascii"):
            fields = line.split()
            if fields[6] == "/robots.txt":
                robots.add(fields[0])
    return robots


def find_all_robots(logdir):
    files = glob.glob(logdir + "/*.log.gz")
    all_robots = set()
    with futures.ProcessPoolExecutor() as pool:
        for robots in pool.map(find_robots, files):
            all_robots.update(robots)
    return all_robots


class ActorExit(Exception):
    pass


class Actor:
    def __init__(self):
        self._mailbox = Queue()

    def send(self, msg):
        self._mailbox.put(msg)

    def recv(self):
        msg = self._mailbox.get()
        if msg is ActorExit:
            raise ActorExit()
        return msg

    def close(self):
        self.send(ActorExit)

    def start(self):
        self._terminated = Event()
        t = Thread(target=self._bootstrap)
        t.daemon = True
        t.start()

    def _bootstrap(self):
        try:
            self.run()
        except ActorExit:
            pass
        finally:
            self._terminated.set()

    def join(self):
        self._terminated.wait()

    def run(self):
        while True:
            msg = self.recv()


class PrintActor(Actor):
    def run(self):
        while True:
            msg = self.recv()
            print("Got:", msg)


if __name__ == '__main__':
    p = PrintActor()
    p.start()
    p.send("hello")
    p.send("world")
    p.close()
    p.join()


