import threading
import numpy as np
import networkx as nx
from scipy.optimize import minimize # type: ignore

from Logger import *
from Random import *


class Graph:

    '''
    Graph class used to represent the network of
    devices participating in training.
    '''

    def __init__(
        self, 
        logger: Logger, 
        random: Random
    ) -> None:

        '''
        random - random number generator used to create and decompose the graph.
        '''
        
        self.logger: Logger = logger
        self.random: Random = random

        '''
        Probabilities are the probability of a subgraph activating.
        When activated, the devices in the subgraph share the models
        to their respective neighbors in the subgraph.
        '''

        self.G: nx.Graph = nx.empty_graph() # type: ignore
        self.subgraphs: list [set [tuple [str, str]]] = []
        self.probabilities: tuple [float] = tuple()

        '''
        The label_map is to map the ip address of the devices to integers.
        Required for computing the probabilities.
        '''

        self.num_nodes: int = 0
        self.label_map: dict [str, int] = dict()

        '''
        We need a lock for the graph, so that it is merged with
        new graphs correctly.
        '''

        self.received: list [nx.Graph] = []
        self.lock: threading.Lock = threading.Lock()

    
    def add_edge(self, node1: str, node2: str) -> None:

        '''
        Adds an edge to the graph if it is not a duplicate and 
        not an edge from a node to itself.
        '''

        if node1==node2: return
        if (node1, node2) in self.G.edges: return

        if node1 not in self.label_map:
            self.label_map[node1] = self.num_nodes
            self.num_nodes += 1

        if node2 not in self.label_map:
            self.label_map[node2] = self.num_nodes
            self.num_nodes += 1

        self.G.add_edge(node1, node2)

        if self.logger: self.logger.log(
            "Graph.add_edge",
            "Added edge from %s to %s",
            node1, node2
        )

    
    def equals(self, other: nx.Graph) -> bool:

        '''
        Returns ture if two graphs have same nodes and edges.
        '''

        return nx.utils.misc.graphs_equal(self.G, other)
    

    def degree(self) -> float:

        '''
        Returns the average node degree.
        '''
        
        return (
            0.5 * 
            sum(map(lambda x:x[1], self.G.degree())) /  # type: ignore
            float(self.num_nodes)
        )
    

    def merge(self):

        '''
        Merges the graph with others. The edges don't have to be disjoint.
        '''

        with self.lock:

            n = len(self.received)
            if n <= 0: return

            if self.logger: self.logger.log(
                "Graph.merge",
                "Merging %d graphs.",
                n
            )

            G = nx.operators.all.compose_all(
                self.received # type: ignore
            )

            for (n1, n2) in G.edges:
                self.add_edge(n1, n2)

            self.received = []

    
    def is_empty(self) -> bool:

        '''
        Returns true if the graph has no edges.
        '''

        return nx.function.is_empty(self.G)

    
    def truncate(self):

        '''
        Replace the graph with an empy graph.
        '''

        self.G = nx.empty_graph() # type: ignore

    
    def neighbors(self, node: str) -> set [str]:

        '''
        Returns a set of adjacent nodes.
        '''

        return set(self.G.neighbors(node))


    def create_random(
        self, 
        n: int = 10, 
        k: int = 4, 
        p: float = 0.33
    ) -> None:
        
        '''
        Creates a random connected graph with the specified nodes and density.
        Labels are typically the ip addresses of the devices that need to be 
        specified before calling this function.
        Used in the simulation if needed.
        '''

        G = (
            self.random.graph if self.random 
            else nx.connected_watts_strogatz_graph(n, k, p)
        )

        self.G = nx.empty_graph() # type: ignore

        '''
        Add the edges from the temporary graph to the self.G
        '''

        for (n1, n2) in G.edges:
            self.add_edge(n1, n2)

        if self.logger: self.logger.log(
            "Graph.create_random",
            "Graph created with %d nodes and %d edges.",
            self.G.number_of_nodes(),
            self.G.number_of_edges()
        )
        

    def decompose(
        self, 
        method: str = "matcha", 
        comm_budget: float = 0.25
    ) -> None:

        '''
        Decomposes the graph into subgraphs with the
        specified method.

        MATCHA  - Decompose the graph into matchings.
        SUBSET  - Random subset of edges in each subgraph.
        SINGLE  - Single edge in each subgraph 
        '''

        if self.logger: self.logger.log(
            "Graph.decompose",
            "Decomposing graph with method: %s.",
            method
        )

        method = method.lower()

        if method == "matcha":

            G: nx.Graph = self.G.copy()
            subgraphs: list [set] = []

            for i in range(
                max(0, self.G.number_of_nodes()-1)
            ):

                '''
                Get maximal matchings of the graph.
                If it's a perfect matching add the edges to a new subgraph.
                Remove the edges from the graph. Repeat until all edges are removed.
                '''

                if G.number_of_edges() == 0: break
                
                matching: set = nx.maximal_matching(G) # type: ignore
                
                if nx.is_perfect_matching(G, matching): # type: ignore
                    G.remove_edges_from(matching)
                    subgraphs.append(matching)
                
                else: break

            '''
            If we couldn't create perfect matchings for all edges,
            Create maximal matchings until all edges are removed.
            '''
            
            safety: int = 0
            limit: int = G.number_of_edges()*2
            while G.number_of_edges() != 0 and safety < limit:

                safety += 1

                matching: set = nx.maximal_matching(G) # type: ignore
                G.remove_edges_from(matching)
                subgraphs.append(matching)

            '''
            Finally, remove any empty subgraphs.
            '''

            self.subgraphs = list(filter(
                lambda x: len(x)>0,
                subgraphs
            ))

        elif method == "subset":

            self.subgraphs = []
            edges: list = list(map(tuple, self.G.edges))

            n = len(edges)
            m = int(round(n*comm_budget))
            
            while n>0:

                if self.random: edges = self.random.shuffle(edges)
                else: np.random.shuffle(edges)

                self.subgraphs.append(set(edges[:m]))
                n -= m

        elif method == "single":

            self.subgraphs = list(
                map(
                    lambda x: {tuple(x)}, 
                    self.G.edges
                )
            )

        total_edges = sum(
            len(subgraph) for subgraph in self.subgraphs
        )

        if self.logger: self.logger.log(
            "Graph.decompose",
            "Created %d subgraphs (total_edges: %d).",
            len(self.subgraphs),
            total_edges
        )

        if self.logger: self.logger.log(
            "Graph.decompose",
            "Subgraphs: \n%s",
            "\n".join(
                (
                    "| subgraph-%d (\\w %d edges): [%s]" % 
                    ( 
                        i, len(subgraph), 
                        ", ".join(map(lambda x:x[0]+"->"+x[1], subgraph))
                    ) 
                )
                for i, subgraph in enumerate(self.subgraphs)
            )
        )


    def compute_probabilities(
        self, 
        method: str = "matcha", 
        comm_budget: float = 0.25
    ) -> None:
        
        '''
        Compute the activation probabilities of each subgraph using
        the specified method, meeting the communication budget. 
        For now, we only support MATCHA.

        MATCHA  - Assign probabilities proportional to the
                  algebraic connectivity of the formed subgraph.
        
        RANDOM  - Random probability.
        '''
        
        if self.logger: self.logger.log(
            "Graph.compute_probabilities",
            "Computing probabilities with method: %s and budget: %.4f.",
            method,
            comm_budget
        )

        '''
        If comm_budget is 0 or 1 we can just set the probabilities
        to 0 or 1 respectively, irrespective of the method.
        '''

        if comm_budget > (1 - (1e-10)):

            self.probabilities = tuple(
                [1.0 for _ in self.subgraphs]
            )

        elif comm_budget < (1e-10):

            self.probabilities = tuple(
                [0.0 for _ in self.subgraphs]
            )
        
        elif method == "matcha":

            '''
            Compute the Laplacian matrices for each subgraph.
            '''

            m = len(self.subgraphs)
            n = self.G.number_of_nodes()

            L = np.zeros((m, n, n))
            for i, subgraph in enumerate(self.subgraphs):

                tmp_G = nx.Graph()
                tmp_G.add_nodes_from(self.G.nodes)
                tmp_G.add_edges_from(subgraph)
                tmp_G = nx.relabel_nodes(tmp_G, self.label_map) # type: ignore

                L[i] = nx.laplacian_matrix( # type: ignore
                        tmp_G,
                        list(range(self.num_nodes))
                    ).todense()

            objective = lambda P: -np.sort(
                np.linalg.eigvalsh(np.einsum('i,ijk->jk', P, L))
            )[1]

            '''
            Probability values are in range [0, 1]
            Add constraint that Sum(Probabilites) <= Cb*|V|
            '''

            bounds = [(0, 1) for i in range(m)]
            constraint1 = {
                'type': 'ineq', 
                'fun': lambda P: (comm_budget*m) - np.sum(P)
            }
            constraints = [constraint1]

            '''
            Compute the probabilites
            '''

            P0 = (
                self.random.seq[:m].copy() * (
                    1.0 / np.linalg.norm(self.random.seq[:m])
                ) if self.random
                else np.random.rand(m)
            )

            result = minimize(
                objective, 
                P0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )

            P = result.x
            self.probabilities = tuple(P)

            Sum_PL = np.einsum('i,ijk->jk', P, L)
            cb = float(sum(P))/float(m)

            if self.logger: self.logger.log(
                "Graph.compute_probabilities",
                "l2: %.4f, Comm ratio: %.4f",
                np.sort(np.linalg.eigvalsh(Sum_PL))[1],
                cb
            )

        elif method == "random":

            m = len(self.subgraphs)
            n = self.G.number_of_nodes()

            '''
            Compute the probabilites
            '''

            P = [
                self.random.rand() if self.random else np.random.rand()
                for _ in range(m)
            ]

            sp = sum(P)
            P = [
                p * comm_budget * m / sp
                for p in P
            ]

            self.probabilities = tuple(P)
            cb = float(sum(P))/float(m)

            if self.logger: self.logger.log(
                "Graph.compute_probabilities",
                "Comm ratio: %.4f",
                cb
            )
            
        if self.logger: self.logger.log(
            "Graph.compute_probabilities",
            "Probabilities: \n%s",
            "\n".join(
                (
                    "| subgraph-%d (\\w %d edges): %.8f" % 
                    ( 
                        i, len(self.subgraphs[i]), 
                        self.probabilities[i]
                    ) 
                )
                for i in range(len(self.subgraphs))
            )
        )

        
    def get_activations(self, subset = -1) -> tuple [tuple [str, str], ...]:
        
        '''
        Once the activation probabilites are computed,
        we can get which subgraphs are activated at each training iteration.
        This function will return the tuple of edges that are activated.
        '''

        '''
        Find which subgraphs are activated.
        '''

        n = len(self.subgraphs)
        flags = [0 for _ in range(n)]

        if subset > 0:

            while subset > 0:

                x = self.random.randint(0, n-1)
                while flags[x]: x = self.random.randint(0, n-1)
                flags[x] = 1

                subset -= 1

        else:

            for i, p in enumerate(self.probabilities):

                if (
                    self.random.rand() if self.random 
                    else np.random.rand()
                ) <= p:
                    
                    flags[i] = 1

        '''
        Get the edges from the activated subgraphs.
        '''

        activations = tuple(
            edge for i in range(n)
            for edge in self.subgraphs[i] 
            if flags[i]
        )

        if self.logger: self.logger.log(
            "Graph.get_activations",
            "Activating %d subgraphs with %d edges.",
            len([f for f in flags if f]), 
            len(activations)
        )

        if len(activations) > 0:

            if self.logger: self.logger.log(
                "Graph.get_activations",
                "Activations: \n%s",
                "\n".join(
                    (
                        "| subgraph-%d (\\w %.8f probability): [%s]" % 
                        ( 
                            i, self.probabilities[i],
                            ", ".join(
                                map(lambda x:x[0]+"->"+x[1], self.subgraphs[i])
                            )
                        ) 
                    )
                    for i in range(n)
                    if flags[i]
                )
            )

        return activations


if __name__ == "__main__":

    '''
    Testing script
    '''

    comm_bugdet = 0.25

    graph = Graph(
        None, None # type: ignore
    )

    graph.create_random()

    graph.decompose(
        method = "matcha",
        comm_budget = comm_bugdet
    )

    graph.compute_probabilities(
        method = "matcha",
        comm_budget = comm_bugdet
    )

    for x in graph.get_activations():

        print(x)
    
   