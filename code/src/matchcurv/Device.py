from tqdm import tqdm
import json
import time
import pickle
import os

from Logger import *
from Packet import *
from Comm import *
from Graph import *
from Model import *
from Database import *
from Random import *


class Device:

    '''
    The Device class represents the Edge device.
    It needs an IP and two free ports for TCP and UDP communication.
    '''

    def __init__(
        self,
        config: str,
        id: int = -1,
        start_time: float = 0
    ) -> None:
        
        '''
        config      - The attributes of the device are derived from a config file.
        id          - In a simulation, we also have to pass an 'id' argument.
                      This is to create unique behaviour based on the id for creating a
                      real world scenario.
        start_time  - Every device will wait until [start_time + delay] 
                      for other devices to catch up.

        Simulation: new_process(
            "python Device.py path/to/config.json " + str(id) + " " + str(start_time)
        )

        Otherwise: > python Device.py path/to/config.json -1 $current timestamp + delay$
        '''

        self.config: dict
        with open(config, "r") as config_file:
            self.config = json.load(config_file)

        '''
        For the sake of simulation that's run on a single machine,
        we need to write our own pseudo random generator.
        '''

        self.random: Random = Random(
            self.config["random_preset"],
            self.config["graph_preset"],
            self.config["model_preset"]
        )

        '''
        The Comm class needs to know if the address is local or not.
        '''

        self.id = id
        self.sim: bool = self.config["is_simulation"] == "True"
        self.start_time = start_time + self.config["sync_delay"]
        
        self.ip: str = (
            ("127.0.0.%d" % (self.id+1)) if self.sim 
            else self.config["ip"]
        )

        self.logger: Logger = (             # type: ignore
            None if self.config["ignore_logger"] == "True"
            else Logger(
                location = self.config["logs_location"],
                prefix = self.ip.replace('.', '-'),
                stdout = self.config["stdout"] == "True" or self.id==0
            )
        )

        '''
        straggler or not? 

        Randomly select provided proportion of stragglers from the devices.
        Make sure the graph is connected and Device 0 is not a straggler.
        '''

        self.straggler = False

        if self.sim and self.random.graph:

            temp = None
            stragglers = []
            num_stragglers = int(round(
                self.config["stragglers"] * 
                self.random.graph.number_of_nodes()
            ))

            if num_stragglers > 0:

                nodes = list(self.random.graph.nodes())

                for _ in range(10000):

                    stragglers = self.random.choices(
                        nodes, num_stragglers
                    )

                    if "127.0.0.1" in stragglers: continue

                    temp = self.random.graph.copy()
                    temp.remove_nodes_from(stragglers)

                    if nx.is_connected(temp): break
                    
                self.straggler = self.ip in stragglers

        '''
        If the device is a straggler and the others are set to ignore it,
        we might as well stop it.
        '''

        self.stopped = False
        self.comm: Comm = Comm(
            ip = self.ip, 
            tcp_port = self.config["tcp_port"], 
            udp_port = self.config["udp_port"],
            logger = self.logger, # type: ignore
            sim = self.sim
        )

        if (
            self.straggler and 
            self.config["mode"] == "sync" and 
            self.config["num_seconds"] < 0
        ): 
            self.stop(dump = False)
            return

        '''
        We create the empty graph initially and populate it
        once we started the training.
        '''

        self.num_activations: list [int] = []
        self.neighbors: set [str] = set()
        self.graph: Graph = Graph(
            logger = self.logger,   # type: ignore
            random = self.random
        )

        '''
        And then create a ML model according to the config.
        If we are training synchronously, we merge the model waits
        from other devices at the end of the round. If not, they are
        merged as they are received.
        '''

        self.model: Model
        
        if self.config["model"] == "MLP":

            self.model = MLP(
                self.logger, # type: ignore
                self.random,
                self.config["mode"] == "sync"
            )

        elif self.config["model"] == "LeNet5":

            self.model = LeNet5(
                self.logger, # type: ignore
                self.random,
                self.config["mode"] == "sync"
            )

        self.model.create(
            self.config["input_shape"],
            self.config["num_outputs"], 
            self.config["learning_rate"],
            self.config["l2_term"],
            self.config["prox_term"],
            self.config["curv_term"],
            *self.config["model_args"]
        )

        '''
        We need to wait until our immediate neighbors reply (sometimes).
        The roll_sheet keeps track of who replied.
        '''

        self.roll_sheet: dict [str, set [str]] = dict()
        self.sheet_lock: threading.Lock = threading.Lock()
        self.vibing: bool = False

        '''
        Finally. connect to the cloud for feeding values to the dashboard.
        Only necessary for monitoring the training.
        '''

        self.database: Database =  (          # type: ignore
            None if self.config["ignore_database"] == "True"
            else Database(
                self.logger,                  # type: ignore
                self.config["certificate"],
                self.config["database_url"],
                self.ip.replace('.', '_')
            )
        )                                       

        try: 

            if self.database: 
                self.database.connect()

        except:

            if self.logger: self.logger.log(
                "Device.init",
                "Failed to connect to the database."
            )
                
            try: self.database.disconnect() # type: ignore
            except: pass

            self.database = None # type: ignore

    
    def start(self) -> None:

        '''
        On creation of Device object, we don't start listening
        or responding to others. We have to call the start method for that.
        '''

        self.comm.start(self.packet_handler)

        if self.logger: self.logger.log(
            "Device.start",
            "Device started listening and responding."
        )

        '''
        For now we randomly decide if we want to start the
        FL training in this device.
        '''

        if self.sim or not (self.random.randint(0, 9) and self.id): 
            self.start_the_music()


    def stop(self, dump = True) -> None:

        '''
        At the end we need to close the comm sockets and other clean up.
        '''

        if self.logger: self.logger.log(
            "Device.stop",
            "Device will stop listening and responding."
        )
            
        try: 

            self.vibing = False
            self.comm.stop()

        except: pass

        self.stopped = True
            
        try:

            '''
            Dump the results.
            '''

            if dump: 

                location = self.config["results_location"]

                if not os.path.exists(location): 
                    os.makedirs(location)

                with open(
                    location + self.ip.replace('.', '_') + ".b", "wb"
                ) as file:
                    
                    result = (
                        self.num_activations,
                        self.model.train_acc,
                        self.model.train_loss,
                        self.model.test_acc,
                        self.model.test_loss
                    )

                    pickle.dump(result, file)

                print("%s dumped results." % self.ip)

        except: pass

        try:

            if self.database: self.database.disconnect()

        except: pass

        '''
        Let the simulation parent process know we are finished.
        '''

        time.sleep(self.config["sync_delay"])

        try:

            self.send(
                "Adios!", 
                pickle.dumps(self.id), 
                "127.0.0.250"
            )

        except: pass


    def send(
        self, 
        packet_type: str, 
        payload: bytes, 
        dest_ip: str
    ) -> bool:

        '''
        To send a tcp packet to the destination using its IP 
        and assume that it's listening at the same port 
        as the self.tcp_port.
        '''

        packet: Packet = Packet(
            sender = (self.comm.ip, 0),
            destination = (dest_ip, self.comm.tcp_port),
            type = packet_type,
            payload = payload
        )

        sent: bool = self.comm.send(packet)

        return sent
    

    def broadcast(self, packet_type: str, payload: bytes) -> bool:

        '''
        To broadcast a udp packet and assume that 
        the devices are listening at the same port 
        as the self.udp_port. 
        '''

        sent = False

        '''
        In the simulation, broadcasting doesn't work.
        So just send tcp packets addressed to neighbors.
        '''

        if self.sim:

            sent = True
            for neighbor in self.random.graph.neighbors(self.ip):

                packet = Packet(
                    sender = (self.comm.ip, 0),
                    destination = (neighbor, self.comm.udp_port),
                    type = packet_type,
                    payload = payload
                )

                sent = sent and self.comm.broadcast(packet)

        else:

            '''
            Outside the simulation, broadcasting works :-D.
            '''

            packet = Packet(
                sender = (self.comm.ip, 0),
                destination = ("<broadcast>", self.comm.udp_port),
                type = packet_type,
                payload = payload
            )

            sent = self.comm.broadcast(packet)

        return sent
    

    def roll_call(
        self, 
        key: str, 
        participants: set [str], 
        duration: float,
    ) -> set [str]:

        '''
        A blocking call to wait until we receive packets from
        the devices we are waiting on.

        key - packet type for which we are waiting
        participants - Ip addresses from which we are expecting
        duration - duration for which we wait at most.
        '''

        if self.logger: self.logger.log(
            "Device.roll_call",
            "Waiting for: %s",
            ', '.join(participants)
        )

        dt: float = max(1, float(duration)/300.0)

        total: int = len(participants)
        waited: float = 0
        present: set [str] = set()

        '''
        We wait until all participants respond or until timeout.
        '''

        while waited < duration and len(participants) > 0:

            with self.sheet_lock:

                try:

                    present = present.union(
                        participants.intersection(
                            self.roll_sheet[key]
                        )
                    )

                    participants = participants.difference(
                        self.roll_sheet[key]
                    )

                    self.roll_sheet[key].clear()
                
                except: pass

            time.sleep(dt)
            waited += dt

        '''Empyt the sheet to be reused again.'''

        with self.sheet_lock:
            
            try: self.roll_sheet.pop(key)
            except: pass

        if self.logger: self.logger.log(
            "Device.roll_call",
            "%d of %d devices replied.",
            len(present),
            total
        )

        if len(participants)>0:
            
            if self.logger: self.logger.log(
                "Device.roll_call",
                "Didn't hear from %s.",
                ', '.join(participants)
            )

        return present
    

    def start_the_music(self):

        '''
        self.vibing is set to true so we don't run
        this function in multiple threads.
        '''

        if self.vibing: return
        self.vibing = True

        if self.logger: self.logger.log(
            "Device.start_the_music",
            "ybin'...."
        )
            
        '''Dump the config to stdout and logs.'''

        for key in self.config:

            if self.logger: self.logger.log(
                "Device.start_the_music",
                "%s: %s",
                key, str(self.config[key])
            )
                
        '''
        Propose training to other devices.
        And find who our immediate neighbors are.
        '''

        self.broadcast(
            packet_type = "Whammo!",
            payload = "Awaken, my masters!".encode()
        )

        '''
        In the sim we already know our neigbhors,
        So don't bother waiting.
        '''

        if self.sim or self.random.graph: self.graph.create_random()
        else: time.sleep(100)

        '''
        If no one in our vicinity is interested, we stop.
        '''

        self.neighbors = self.graph.neighbors(self.ip)

        if len(self.neighbors) <= 0:

            if self.logger: self.logger.log(
                "Device.start_the_music",
                "None reachable. Stopping now."
            )

            self.stop()
            return

        if self.logger: self.logger.log(
            "Device.start_the_music",
            "Neighbors: [%s]",
            ", ".join(self.neighbors)
        )
                
        '''
        Once we figured out who our neighbors are,
        We request info about who our neighbors neighbors are.
        We do this untill all our neighbors return the same topology.
        Resulting in a concensus on the topology of entire network.
        '''

        if not (self.sim or self.random.graph):

            '''
            Agree on the topology of the training network.
            Start by sending the graph we have to our neighbors.
            '''

            for i in range(int(1e3)):

                if not self.vibing: 
                    
                    self.stop(False)
                    return

                for neighbor in self.neighbors:

                    self.send(
                        packet_type = "Network Update",
                        payload = pickle.dumps(self.graph.G),
                        dest_ip = neighbor
                    )

                '''
                Wait until we receive graphs from neighbors.
                '''

                self.roll_call(
                    key = "ACK: Network Update",
                    participants = self.neighbors,
                    duration = 100
                )

                '''
                Check if every neighbor returned the same graph.
                If true, we break the loop.
                '''

                consensus = True

                with self.graph.lock:
                
                    for received_graph in self.graph.received:
                        consensus &= self.graph.equals(received_graph)

                if consensus:
                    
                    if self.logger: self.logger.log(
                        "Device.start_the_music",
                        "Reached concensus in %d iterations.",
                        i+1
                    )

                    break

                '''
                If there isn't consensus, we merge the all received graphs
                and repeat.
                '''

                self.graph.merge()

        '''
        If no one in our vicinity is interested, we stop.
        '''

        if self.graph.is_empty(): 
            
            if self.logger: self.logger.log(
                "Device.start_the_music",
                "Empty graph. Shutting down."
            )

            self.stop()
            return
        
        '''
        Else, we run the Matcha algorithm.
        '''

        self.graph.decompose(
            method = self.config["decomposition"],
            comm_budget = self.config["comm_budget"]
        )

        self.graph.compute_probabilities(
            method = self.config["activations"],
            comm_budget = self.config["comm_budget"]
        )

        '''
        Connect to the database and set the values for a new session.
        '''

        if self.database: 
            self.database.reset()

        '''
        Then, start the training, by first loading the dataset.
        In the simulation we specify what portion of data 
        to use in training.
        '''

        train_files = [
            self.config["train_files"] + filename
            for filename in os.listdir(self.config["train_files"])
        ]

        test_files = [
            self.config["test_files"] + filename
            for filename in os.listdir(self.config["test_files"])
        ]

        n_train = len(train_files)
        l_start, l_end = 0, n_train
        s_start, s_end = 0.0, 1.0

        if self.sim:

            ld = self.config["label_distribution"]
            sd = self.config["sample_distribution"]

            l_start = (self.id * ld) % n_train
            l_end = l_start + ld
            s_start = (self.id * sd) % 1
            s_end = s_start + sd
        
        '''
        Load our portion of data in the simulation.
        Otherwise, load all of the local data.
        '''

        self.model.preheat(
            train_files = train_files,
            test_files = test_files,
            label_distribution = (l_start, l_end),
            sample_distribution = (s_start, s_end)
        )

        '''
        Synchronize with other devices before starting training
        '''

        t = self.start_time
        t_ = time.time()
        while t_ < t:
            time.sleep(1)
            t_ = time.time()
        
        _tqdm = lambda x: tqdm(x) if not self.id else x
        self.model.tqdm = lambda x: tqdm(x) if not self.id else x

        num_epochs = self.config["num_epochs"]
        num_rounds = self.config["num_rounds"]

        with self.model.lock:

            for r in range(num_rounds):

                if r not in self.model.model_queue:
                    self.model.model_queue[r] = {}

        '''
        Training loop
        '''

        for r in _tqdm(range(num_rounds)):

            if not self.vibing: 
                
                self.stop()
                return
            
            if self.logger: self.logger.log(
                "Device.start_the_music",
                "Training round {%d} of {%d}",
                r+1, num_rounds
            )

            if self.database: self.database.set(
                path = "round", 
                data = r
            )
            
            '''
            If a straggler we stagnate training for a few seconds.
            '''

            if self.straggler: 
                time.sleep(self.config["lag"])
                
            '''
            Train the model.
            '''
            
            self.model.train(
                # num_seconds = self.config["num_seconds"] - (
                #     self.config["lag"] if self.straggler else 0
                # ),
                num_seconds = -1,
                # num_epochs = num_epochs,
                num_epochs = num_epochs // (2 if self.straggler else 1),
                batch_size = self.config["batch_size"]
            )

            '''
            Find who to share models with from the activations.
            '''

            activations = self.graph.get_activations()
            recepients = set()

            for node1, node2 in activations:

                if node1==self.ip: recepients.add(node2)
                elif node2==self.ip: recepients.add(node1)

            if self.logger: self.logger.log(
                "Device.start_the_music",
                "Sharing weights with {%d}: [%s]",
                len(recepients),
                ", ".join(recepients)
            )

            self.num_activations.append(len(recepients))
            if self.database: 

                self.database.set(
                    path = "shares", 
                    data = sum(self.num_activations)
                )

            if self.logger: self.logger.log(
                "Device.start_the_music",
                "Number of shares: %d",
                self.num_activations[-1]
            )
                
            '''
            Compute Fisher information matrix and
            Share fim and model weights according to activations.
            '''
                
            if self.model.curv_term:
                self.model.compute_fim()
            
            expected_senders = set()
            for neighbor in recepients:
                
                with self.model.lock:

                    if self.send(
                        packet_type = "Model Update",
                        payload = pickle.dumps(
                            (
                                r, 
                                self.model.fim,
                                self.model.model.weights
                            )
                        ),
                        dest_ip = neighbor
                    ):
                        expected_senders.add(neighbor)

            '''
            If we are synchronizing training,
            we wait until others finish their training round
            and merge the models from them with ours.
            '''

            if self.model.sync:

                with self.model.lock:

                    expected_senders = expected_senders.difference(
                        list(
                            self.model.model_queue[r].keys()
                        )
                    )

                self.roll_call(
                    key = "Model Update",
                    participants = expected_senders,
                    duration = 600
                )

                self.model.merge(r)

            '''
            Evaluate the model.
            '''

            self.model.test()
            
            '''
            Send the values to the cloud for the dashboard.
            '''

            if self.database: 

                self.database.update(
                    path = "local_acc",
                    data = {r: self.model.train_acc[-1]}
                )

                self.database.update(
                    path = "local_loss",
                    data = {r: self.model.train_loss[-1]}
                )

                self.database.update(
                    path = "global_acc",
                    data = {r: self.model.test_acc[-1]}
                )

                self.database.update(
                    path = "global_loss",
                    data = {r: self.model.test_loss[-1]}
                )

        '''
        Tidy up things at the end.
        '''

        if self.database: self.database.set(
            path = "round", 
            data = num_rounds
        )

        if self.logger: self.logger.log(
            "Device.start_the_music",
            "Training Finished."
        )

        self.stop()


    def packet_handler(self, packet: Packet) -> None:

        '''
        This function handles all incoming packets
        and takes appropriate actions.
        '''

        if packet.type[:3] == "ACK":

            '''
            If it's an acknowledgement packet,
            we need to let the roll_sheet know about it.
            '''

            with self.sheet_lock:

                if packet.type not in self.roll_sheet:
                    self.roll_sheet[packet.type] = {packet.sender[0]}

                else:
                    self.roll_sheet[packet.type].add(packet.sender[0])

        if packet.type == "Whammo!":

            '''
            If it's a proposal for training, 
            if we are not already training,
            we start the training.
            '''

            if not self.vibing: 
                
                threading.Thread(
                    target = self.start_the_music
                ).start() 

            '''
            By sending this acknowledgement, we let the sender know
            that we are training and a neighbor of its.
            '''

            self.send(
                packet_type = "ACK: " + packet.type,
                payload = "".encode(),
                dest_ip = packet.sender[0]
            )

        elif packet.type == "Network Update":

            '''
            If someone is sending a network update,
            we use it to update our graph and send an acknowledgement.
            '''

            with self.graph.lock:

                self.graph.received.append(
                    pickle.loads(packet.payload)
                )

                self.send(
                    packet_type = "ACK: " + packet.type,
                    payload = pickle.dumps(self.graph.G),
                    dest_ip = packet.sender[0]
                )

        elif packet.type == "Model Update":

            '''
            Just like a network update, we store the model updates
            - to be merged later. If we are sync mode, we need to 
            let the rollsheet to know we have received the model update
            for the address of the sender.
            '''

            with self.model.lock:

                r, fim, weights = pickle.loads(packet.payload)

                if self.model.curv_term:

                    self.model.received_fims[packet.sender[0]] = fim
                    self.model.received_weights[packet.sender[0]] = weights

                try: 
                    
                    self.model.model_queue[r][packet.sender[0]] = weights

                except:

                    self.model.model_queue[r] = {}
                    self.model.model_queue[r][packet.sender[0]] = weights

            with self.sheet_lock:

                if packet.type not in self.roll_sheet:

                    self.roll_sheet[packet.type] = {
                        packet.sender[0]
                    }

                else:

                    self.roll_sheet[packet.type].add(
                        packet.sender[0]
                    )

        elif packet.type == "Stop":

            '''
            The parent process (Simulation.py) wants to interrupt the process,
            it can send "Stop" packet.
            '''

            self.stop()
            return
            

if __name__ == "__main__":

    '''
    This script is executed for running the device.
    '''

    import sys

    device = Device(
        config = sys.argv[1],
        id = int(sys.argv[2]),
        start_time = float(sys.argv[3])
    )

    if not device.stopped:

        try: device.start()
        except Exception as e: 

            if device.logger: device.logger.log(
                "Main",
                "Exception: %s",
                str(e)
            )
            
            print("\n========================================")
            print(device.ip, ":", str(e))
            print("========================================\n")

            device.stop()
    