# common_mp.py
import socket
import struct
import io
import torch


def send_msg(conn, obj):
    """
    Gửi một Python object (tensor, dict, ...) qua socket.
    Dùng torch.save để serialize.
    """
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()

    # Gửi độ dài trước (4 bytes, big endian)
    conn.sendall(struct.pack('>I', len(data)))
    conn.sendall(data)


def recv_msg(conn, map_location=None):
    """
    Nhận một Python object qua socket.
    Nếu map_location != None, dùng cho tensor (map sang CPU/GPU mong muốn).
    """
    header = conn.recv(4)
    if not header:
        return None
    (size,) = struct.unpack('>I', header)

    data = b''
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            break
        data += chunk

    if not data:
        return None

    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location=map_location)
