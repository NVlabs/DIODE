import os
import sys

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


#based on https://groups.google.com/forum/#!topic/comp.lang.python/0lqfVgjkc68

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        bufsize = 1
        self.log = open(filename, "w", buffering=bufsize)

    def delink(self):
        self.log.close()
        self.log = open('foo', "w")
#        self.write = self.writeTerminalOnly

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass