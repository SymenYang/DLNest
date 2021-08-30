from DLNest.TornadoServer.Server import DLNestServer


if __name__ == "__main__":
    import sys
    if sys.path[0] != '':
        sys.path[0] = ''
    server = DLNestServer()
    server.start()