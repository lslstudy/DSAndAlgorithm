# -*- coding: utf-8 -*-

""" sync blocked client
"""

import socket


HOST = "localhost"
PORT = 3268

# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# client.connect((HOST, PORT))
#
# while True:
#     try:
#         data = input("please in put: ")
#         if data.strip() == "exit":
#             break
#     except Exception as e:
#         break
#
#     client.send(bytes(data, encoding="utf-8"))
#     result = client.recv(1024)
#     print(f"client receive data={result}")


if __name__ == '__main__':
    from functools import partial

    def add(*args):
        return sum(args)

    add_100 = partial(add, 100)
    print(add_100(10))

    import tornado
    print(tornado.version)
    # import tornado.ioloop
    # tornado.ioloop.IOLoop.current().start()
