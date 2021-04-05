class FakeSubprocess:
    def __init__(self,alive):
        self.alive = alive
    
    def is_alive(self):
        return self.alive