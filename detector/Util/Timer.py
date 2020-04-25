import time

class Timer:

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time

    def print_elapsed(self, timed_object="", verbose=1):
        # verbose is included here to save if clauses from the clients
        if verbose > 1:
            print("{0} took {1:.3f} seconds".format(timed_object, self.elapsed()))
        self.reset()
