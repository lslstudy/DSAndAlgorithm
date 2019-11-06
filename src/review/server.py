# -*- coding: utf-8 -*-


import socket
import io
import tornado


HOST = "localhost"
PORT = 3268

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.bind((HOST, PORT))

server.listen(5)

print("waiting for connecting")


while True:
    conn, addr = server.accept()
    print(f"connected by {addr}")

    buffer = io.StringIO()
    while True:
        data = conn.recv(1024)
        if data:
            data = str(data, encoding="utf-8")
            print(f"server receive data={data}")
            buffer.write(data)
            conn.sendall(bytes("Hello", encoding="utf-8"))
        else:
            break
    print(f"get all data {buffer.getvalue()}")
    buffer.close()
    conn.close()
    print(f"connection from {addr} closed")
