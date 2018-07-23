from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
import mmap

class ProtobufWriter:
    def __init__(self, filename):
        self.__file = open(filename, 'wb')

    def insert(self, message):
        size = message.ByteSize()
        self.__file.write(_VarintBytes(size))
        self.__file.write(message.SerializeToString())
    
    def close(self):
        self.__file.close()

class ProtobufReader:
    def __init__(self, filename):
        self.__file = open(filename, 'r+')
        self.__buffer = mmap.mmap(self.__file.fileno(), 0)
        self.__buffer_len = len(self.__buffer)
        self.__pos = 0

    def close(self):
        self.__file.close()

    def next(self, proto):
        if self.__pos >= self.__buffer_len:
            self.__file.close()
            return None
        msg_len, new_pos = _DecodeVarint32(self.__buffer, self.__pos)
        self.__pos = new_pos
        msg_buffer = self.__buffer[self.__pos:self.__pos + msg_len]
        self.__pos += msg_len
        proto.ParseFromString(msg_buffer)
        return proto