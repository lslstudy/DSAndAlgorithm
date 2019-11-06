# -*- coding: utf-8 -*-

from collections import deque
from select import select


class YieldEvent(object):

    def handle_yield(self, sched, task):
        pass

    def handle_resume(self, sched, task):
        pass


class Scheduler(object):
    def __init__(self):
        self._numtasks = 0
        self._ready = deque()
        self._read_waitings = {}
        self._write_waitings = {}

    def _iopoll(self):
        rset, wset, eset = select(self._read_waitings, self._write_waitings, [])

        for r in rset:
            evt, task = self._read_waitings.pop(r)
            evt.handle_resume(self, task)

        for w in wset:
            evt, task = self._write_waitings.pop(w)
            evt.handle_resume(self, task)

    def new(self, task):
        self._ready.append((task, None))
        self._numtasks += 1

    def add_ready(self, task, msg=None):
        self._ready.append((task, msg))

    def _read_wait(self, fileno, evt, task):
        self._read_waitings[fileno] = (evt, task)

    def _write_wait(self, fileno, evt, task):
        self._write_waitings[fileno] = (evt, task)

    def run(self):
        while self._numtasks:
            if not self._ready:
                self._iopoll()
            task, msg = self._ready.popleft()
            try:
                r = task.send(msg)
                if isinstance(r, YieldEvent):
                    r.handle_yield(self, task)
                else:
                    raise RuntimeError("unrecognized yield event")
            except StopIteration:
                self._numtasks -= 1


class ReadSocket(YieldEvent):
    def __init__(self, sock, nbytes):
        self.sock = sock
        self.nbytes = nbytes

    def handle_yield(self, sched, task):
        sched._read_wait(self.sock.fileno(), self, task)

    def handle_resume(self, sched, task):
        data = self.sock.recv(self.nbytes)
        sched.add_ready(task, data)


class WriteSocket(YieldEvent):

    def __init__(self, sock, data):
        self.sock = sock
        self.data = data

    def handle_yield(self, sched, task):
        sched._write_wait(self.sock.fileno(), self, task)

    def handle_resume(self, sched, task):
        nsent = self.sock.send(self.data)
        sched.add_ready(task, nsent)


class AcceptSocket(YieldEvent):
    def __init__(self, sock):
        self.sock = sock

    def handle_yield(self, sched, task):
        sched._read_wait(self.sock.fileno(), self, task)

    def handle_resume(self, sched, task):
        r = self.sock.accept()
        sched.add_ready(task, r)


class Socket(object):
    def __init__(self, sock):
        self._sock = sock

    def recv(self, maxbytes):
        return ReadSocket(self._sock, maxbytes)

    def send(self, data):
        return WriteSocket(self._sock, data)

    def accept(self):
        return AcceptSocket(self._sock)

    def __getattr__(self, name):
        return getattr(self._sock, name)


if __name__ == '__main__':
    # from socket import socket, AF_INET, SOCK_STREAM
    #
    # def readline(sock):
    #     chars = []
    #     while True:
    #         c = yield sock.recv(1)
    #         if not c:
    #             break
    #         chars.append(c)
    #         if c == b"\n":
    #             break
    #     return b"".join(chars)
    #
    # class EchoServer(object):
    #     def __init__(self, addr, sched):
    #         self.sched = sched
    #         sched.new(self.server_loop(addr))
    #
    #     def server_loop(self, addr):
    #         s = Socket(socket(AF_INET, SOCK_STREAM))
    #         s.bind(addr)
    #         s.listen(5)
    #         while True:
    #             c, a = yield s.accept()
    #             print("Got connection from", a)
    #             self.sched.new(self.client_handler(Socket(c)))
    #
    #     def client_handler(self, client):
    #         while True:
    #             line = yield from readline(client)
    #             if not line:
    #                 break
    #             line = b"Got:" + line
    #             while line:
    #                 nset = yield client.send(line)
    #                 line = line[nset:]
    #         client.close()
    #         print("client closed")
    # sched = Scheduler()
    # EchoServer(("localhost", 16000), sched)
    # sched.run()

    from functools import partial

    def my_partial(a, b, c, d):
        print(a, b, c, d)

    test1 = partial(my_partial, 1, 2)
    test1(3, 4)


