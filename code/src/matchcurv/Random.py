import pickle
from typing import Any

class Random:

    '''
    A pseudo "random" generator used in simulations.
    '''

    def __init__(
        self, 
        random_preset: str, 
        graph_preset: str,
        model_preset: str,
    ) -> None:

        '''
        - random preset stores a list of randomly generated numbers.
        - graph preset contains a random graph and 2d coords of nodes.
        - model preset contains model weights.
        '''

        self.seq: Any = None
        try:
            with open(random_preset, "rb") as file:
                self.seq = pickle.load(file)
        except: pass
        
        self.graph: Any = None
        self.pos: Any = None
        try:
            with open(graph_preset, "rb") as file:
                self.graph, self.pos = pickle.load(file)
        except: pass

        self.model: Any = None
        try:
            with open(model_preset, "rb") as file:
                self.model = pickle.load(file)
        except: pass
        
        self.counter = 0


    def rand(self) -> float:

        '''
        returns a random float in [0, 1].
        '''

        self.counter += 1
        if self.counter > self.seq.shape[0]: self.counter = 0
        if self.counter < 0: self.counter = 0

        return self.seq[self.counter]
    

    def randint(self, a, b) -> int:

        '''
        returns a random int in [a, b].
        '''

        return int(
            round(self.rand()*(b-a) + a)
        )


    def shuffle(self, arr: list) -> list:

        '''
        returns a shuffled list.
        '''

        res = []
        n = len(arr)

        while n > 0:

            i = self.randint(0, n-1)
            res.append(arr.pop(i))
            n -= 1

        return res
    
    def choice(self, X) -> Any:

        '''
        returns a random value from an iterable
        '''

        n = len(X)
        if n<=0: return None

        try: return X[self.randint(0, n-1)]
        except: return None

    def choices(self, X, n) -> Any:

        '''
        returns n random choices from an iterable
        '''

        if n > len(X): return X

        res = set()
        while len(res) < n: 
            res.add(self.choice(X))

        return list(res)
            

if __name__ == "__main__":

    '''Testing script.'''

    random = Random(
        random_preset = "../../_random_presets/random_preset_1.b",
        graph_preset = "", 
        model_preset = ""
    )

    print([round(random.rand(), 2) for _ in range(20)])
    print([random.randint(0, 100) for _ in range(20)])
    print(random.shuffle([i for i in range(20)]))