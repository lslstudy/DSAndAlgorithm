# —*- coding: utf-8 -*-

""" tronado source demo
"""
from typing import Optional, Awaitable

import tornado.web
import tornado.ioloop


class MainHandler(tornado.web.RequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        self.write("hello world")


if __name__ == "__main__":
    app = tornado.web.Application([(r"/", MainHandler)])
    app.listen(port=8099)
    tornado.ioloop.IOLoop.current().start()
