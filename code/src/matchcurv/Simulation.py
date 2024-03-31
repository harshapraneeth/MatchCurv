import subprocess
import time
import sys
import json
import shutil
import os
import pickle
import itertools as iter

from Database import *
from Random import *
from Comm import *


class Simulation:

    '''
    The instance of this class runs the simulation, provided the config file.
    The IP is used by the subprocesses to indicate their end of execution.
    '''

    def __init__(
        self, 
        ip: str, 
        config_location: str
    ) -> None:

        self.config_location = config_location

        self.config = {}
        with open(self.config_location, "r") as file:
            self.config = json.load(file)

        '''
        Connect to the database and reset the values
        '''

        self.database = Database(
            None, # type: ignore
            self.config["certificate"],
            self.config["database_url"],
            ""
        )

        if self.config["ignore_database"] == "False":

            try:

                self.database.connect()
                self.database.ref.set({})
                self.database.ref.child("entries").set({})
                self.database.set(
                    path = "total_rounds",
                    data = self.config["num_rounds"]
                )
                self.database.disconnect()

            except Exception as e: print("Simulation.init: %s" % str(e))

        '''
        Clear the logs if set to True
        '''

        if self.config["clear_logs"] == "True":

            location = self.config["logs_location"]

            if os.path.exists(location):
                shutil.rmtree(location)
            
            os.makedirs(location)

        '''
        Create folders for logs and results
        '''

        location = self.config["results_location"]

        if os.path.exists(location):
            shutil.rmtree(location)

        os.makedirs(location)

        '''
        Create a set of IPs for the subprocesses participating in the training.
        '''

        self.random = Random(
            self.config["random_preset"],
            self.config["graph_preset"],
            self.config["model_preset"]
        )

        self.participants = set(self.random.graph.nodes())
        self.num_devices = len(self.participants)

        self.roll_sheet: dict [str, set [str]] = dict()
        self.sheet_lock: threading.Lock = threading.Lock()

        self.ip = ip
        self.comm = Comm(
            ip = self.ip,
            tcp_port = self.config["tcp_port"],
            udp_port = self.config["udp_port"],
            logger = None, # type: ignore
            sim = True
        )

        self.processes = []

    
    def start(self):

        '''
        Starts the comm instance and the creates the subprocesses
        '''

        self.comm.start(self.packet_handler)

        self.processes = [
            subprocess.Popen(
                [
                    "python", 
                    "Device.py", 
                    self.config_location,
                    str(i), 
                    str(time.time() + self.config["sync_delay"])
                ]
            )
            for i in range(self.num_devices)
        ]


    def roll_call(
        self, 
        key: str, 
        participants: set [str], 
        duration: float,
    ) -> set [str]:

        '''
        A blocking call to wait until we receive packets from
        the devices we are waiting on.
        '''

        dt: float = 10
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

        return present

    
    def stop(self):

        '''
        Kills all the subprocesses.
        '''

        self.comm.stop()
        
        for i in range(self.num_devices):
            try: self.processes[i].kill()
            except Exception as e: print(e)

    
    def packet_handler(self, packet: Packet):

        '''
        Once a subprocess finishes execution, we kill it.
        '''

        if packet.type == "Adios!":

            try:

                x = pickle.loads(packet.payload)
                self.processes[x].terminate()

                try: self.processes[x].wait(timeout = 30)
                except subprocess.TimeoutExpired: self.processes[x].terminate()

                self.processes[x].kill()
                print("============ Device %d Finished ============" % x)

                with self.sheet_lock:

                    if packet.type not in self.roll_sheet:
                        self.roll_sheet[packet.type] = {packet.sender[0]}

                    else:
                        self.roll_sheet[packet.type].add(packet.sender[0])

            except Exception as e: print(e)


if __name__ == "__main__":

    '''
    This script is run for starting the simulation.
    '''

    CONFIG_PATH = "../../configs/"
    RESULTS_PATH = "../../results/"
    LOGS_PATH = "../../logs/"

    '''
    locations of the config files that need to be run
    '''

    locations = [
        "tiny-imagenet/noniid/"
    ]

    '''
    For each location...
    '''

    for ll, location in enumerate(locations):

        print("\n\nRunning location %d of %d\n\n" % (ll+1, len(locations)))

        config_location = CONFIG_PATH + location if len(sys.argv) < 2 else sys.argv[1]
        results_location = RESULTS_PATH + location if len(sys.argv) < 3 else sys.argv[2]
        logs_location = LOGS_PATH + location if len(sys.argv) < 4 else sys.argv[3]
        search_space = {}

        '''
        Read the search space json file if present
        '''

        config_files = os.listdir(config_location)
        if "_search_space.json" in config_files:

            config_files.remove("_search_space.json")
            with open(config_location + "_search_space.json", "r") as file:
                search_space = json.load(file)
            
        space = list(
            iter.product(
                *[
                    [(key, value) for value in search_space[key][0]]
                    for key in search_space
                ]
            )
        )

        labels = list(
            iter.product(
                *[
                    [(key, value) for value in search_space[key][1]]
                    for key in search_space
                ]
            )
        )

        '''
        For each config in a location...
        '''

        m = len(config_files)
        for j, filename in enumerate(config_files):

            print("\n\nRunning config %d of %d\n\n" % (j+1, m))

            config = {}
            with open(config_location + filename, "r") as file:
                config = json.load(file)

            n = len(space)
            if n > 0:

                '''
                For each vector in the search space,

                create a suffix for the result path and 
                run the experiment.
                '''

                for i, (vector, label) in enumerate(zip(space, labels)):

                    config["results_location"] = (
                        results_location + 
                        "config_v_" + 
                        filename[:filename.rfind(".")] + "_n_"
                    )

                    config["logs_location"] = (
                        logs_location + 
                        "config_v_" + 
                        filename[:filename.rfind(".")] + "_n_"
                    )

                    for (key, value), (_, label) in zip(vector, label):

                        config[key] = value
                        config["results_location"] += key + "_v_" + label + "_n_"
                        config["logs_location"] += key + "_v_" + label + "_n_"

                    config["results_location"] = config["results_location"][:-3]
                    config["results_location"] += "/"

                    config["logs_location"] = config["logs_location"][:-3]
                    config["logs_location"] += "/"

                    with open(config_location + filename, "w") as file:
                        json.dump(config, file)

                    print("\n\nRunning vector %d of %d\n\n" % (i+1, n))
                    
                    simulation = Simulation(
                        "127.0.0.250",
                        config_location + filename
                    )

                    simulation.start()

                    simulation.roll_call(
                        key = "Adios!",
                        participants = simulation.participants,
                        duration = 60 * 60 * 24 * 30
                    )

                    simulation.stop()
                    time.sleep(3 * simulation.comm.timeout)

                    print("\n\nFinished vector %d of %d\n\n" % (i+1, n))

            else:

                config["results_location"] = (
                    results_location + 
                    "config_v_" + 
                    filename[:filename.rfind(".")] + "_n_"
                )

                config["logs_location"] = (
                    logs_location + 
                    "config_v_" + 
                    filename[:filename.rfind(".")] + "_n_"
                )

                config["results_location"] = config["results_location"][:-3]
                config["results_location"] += "/"

                config["logs_location"] = config["logs_location"][:-3]
                config["logs_location"] += "/"

                with open(config_location + filename, "w") as file:
                    json.dump(config, file)

                simulation = Simulation(
                    "127.0.0.250",
                    config_location + filename
                )

                simulation.start()

                simulation.roll_call(
                    key = "Adios!",
                    participants = simulation.participants,
                    duration = 60 * 60 * 24 * 30
                )

                simulation.stop()
                time.sleep(3 * simulation.comm.timeout)

            print("\n\nFinished config %d of %d\n\n" % (j+1, m))