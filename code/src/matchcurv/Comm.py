import socket
import pickle
import time
import threading
from typing import Callable

from Packet import *
from Logger import *


class Comm:

    '''
    The Comm class handles all the communication b/w devices.
    
    We use sockets for communicating b/w:

    - (Outside the simulation) Devices in a network
    - (In the simulation) Processes in a device as well.

    For this we need the ip of the device and 
    two seperate ports to listen to tcp and udp packets.
    '''

    def __init__(
        self, 
        ip: str, 
        tcp_port: int, 
        udp_port: int, 
        logger: Logger, 
        sim: bool = False
    ) -> None:

        self.ip = ip
        self.tcp_port = tcp_port
        self.udp_port = udp_port
        self.logger = logger
        self.sim = sim

        '''
        All incoming packets are pushed to this queue and
        then another thread processes them. 
        
        The listening and processing functions lock the queue
        while altering it.
        '''

        self.packets: list [Packet] = []
        self.packet_lock: threading.Lock = threading.Lock()

        '''
        The listening and responding threads periodically check
        this signal and stop when set to true.
        '''

        self.stop_signal: bool = False
        self.timeout: int = 5 # seconds

    
    def send(self, packet: Packet) -> bool:

        '''
        The send method sends a packet using TCP to the destination
        specified in the packet.destination and returns true if successful.
        '''

        sent = False
        exception = None
        
        try:

            with socket.socket(
                socket.AF_INET, 
                socket.SOCK_STREAM
            ) as sock:

                sock.connect(packet.destination)
                sock.sendall(pickle.dumps(packet))
                # send all: sends all data to the connected destination

                sent = True
            
        except Exception as e: exception = e

        if self.logger: self.logger.log(
            "Comm.send",
            "Sending <%s> packet from %s:%d to %s:%d [%s].",
            packet.type,
            *packet.sender,
            *packet.destination,
            "Sent" if sent else (exception if exception else "Failed")
        )
        
        return sent
    

    def broadcast(self, packet: Packet) -> bool:

        '''
        The broadcast method broadcasts a packet using UDP to all of the
        devices that are reachable and returns true if successful.
        Success doesn't mean the packet reached the destination.
        '''

        sent = False
        exception = None

        try:
            
            with socket.socket(
                socket.AF_INET, 
                socket.SOCK_DGRAM
            ) as sock:

                sock.setsockopt(
                    socket.SOL_SOCKET, 
                    socket.SO_BROADCAST, 
                    1
                )

                sock.bind(packet.sender)

                sock.sendto(
                    pickle.dumps(packet), 
                    packet.destination
                )

                sent = True

        except Exception as e: exception = e
        
        if self.logger: self.logger.log(
            "Comm.broadcast",
            "Broadcasting <%s> packet from %s:%d [%s].",
            packet.type,
            *packet.sender,
            "Sent" if sent else (exception if exception else "Failed")
        )
        
        return sent
    

    def recv_tcp_packet(
        self, 
        sender: socket.socket, 
        address: tuple [str, int],
        chunk_size: int
    ) -> None: 

        '''
        Helper function to receive chunks of the tcp packet
        and stitch them together.
        '''

        received = False
        exception = None
        data = b''

        while not self.stop_signal:

            chunk = sender.recv(chunk_size)
            if not chunk: break
            data += chunk

        sender.close()
        
        '''
        Try to decode the packet. 
        If successfull, push the packet to the packet queue.
        Else, log the error.
        '''

        try:

            packet: Packet = pickle.loads(data)

            '''
            If the sender is ourselves, we ignore the packet.
            Else, we push the packet to the queue.
            '''

            if packet.sender[0]==self.ip: return

            with self.packet_lock:
                self.packets.append(packet)

            received = True

            if self.logger: self.logger.log(
                "Comm.recv_tcp_packet",
                "Receiving <%s> packet from %s:%d [Received].",
                packet.type,
                *packet.sender
            )
        
        except Exception as e: exception = e

        if not received or exception:

            if self.logger: self.logger.log(
                "Comm.recv_tcp_packet",
                "Receiving <unkown> packet from %s:%d [%s].",
                *address,
                exception if exception else "Failed"
            )            
        
        
    

    def tcp_listener(self) -> None:

        '''
        This method listens to all incoming TCP packets and
        pushes them into a queue for the handler. 
        It needs the recv_tcp_packet method to stitch the packet chunks.
        '''

        with socket.socket(
            socket.AF_INET, 
            socket.SOCK_STREAM
        ) as sock:

            sock.bind((self.ip, self.tcp_port))
            sock.listen()
            sock.settimeout(self.timeout)
            
            if self.logger: self.logger.log(
                "Comm.tcp_listener",
                "Started listening for TCP packets at port %d.",
                self.tcp_port
            )

            while not self.stop_signal:

                try:
                    
                    sender, address = sock.accept()
                    self.recv_tcp_packet(sender, address, 2048)
                    
                except socket.timeout: pass
    
        if self.logger: self.logger.log(
            "Comm.tcp_listener",
            "Stopped listening for TCP packets at port %d.",
            self.tcp_port
        )


    def recv_udp_packet(
        self, 
        data: bytes, 
        address: tuple [str, int]
    ) -> None: 

        '''
        Helper function to receive udp packets.
        It's not needed since udp packets are received as a whole not in chunks,
        but programmed this way to create parity with tcp listener.
        '''

        received = False
        exception = None
        
        '''
        Try to decode the packet. 
        If successfull, push the packet to the packet queue.
        Else, log the error.
        '''

        try:

            packet: Packet = pickle.loads(data)

            '''
            If the sender is ourselves, we ignore the packet.
            Else, we push the packet to the queue.
            '''

            if packet.sender[0]==self.ip: return

            with self.packet_lock:
                self.packets.append(packet)

            received = True

            if self.logger: self.logger.log(
                "Comm.recv_udp_packet",
                "Receiving <%s> packet from %s:%d [Received].",
                packet.type,
                *packet.sender
            )
        
        except Exception as e: exception = e

        if not received or exception:

            if self.logger: self.logger.log(
                "Comm.recv_udp_packet",
                "Receiving <unkown> packet from %s:%d [%s].",
                *address,
                exception if exception else "Failed"
            )            


    def udp_listener(self) -> None:

        '''
        This method listens to all incoming UDP packets and
        pushes them into a queue for the handler.
        '''

        with socket.socket(
            socket.AF_INET, 
            socket.SOCK_DGRAM
        ) as sock:

            sock.setsockopt(
                socket.SOL_SOCKET, 
                socket.SO_BROADCAST, 
                1
            )
            
            sock.bind((self.ip if self.sim else '', self.udp_port))
            sock.settimeout(self.timeout)

            if self.logger: self.logger.log(
                "Comm.udp_listener",
                "Started listening for UDP packets at port %d.",
                self.udp_port
            )

            while not self.stop_signal:

                try:

                    data, address = sock.recvfrom(2048)
                    self.recv_udp_packet(data, address)
                    
                except socket.timeout: pass

        if self.logger: self.logger.log(
            "Comm.udp_listener",
            "Stopped listening for UDP packets at port %d.",
            self.udp_port
        )


    def responder(self, packet_handler: Callable [[Packet], None]) -> None:

        '''
        This is a helper function like recv_tcp_packet.
        The function is invoked by resposnd function inside a thread,
        to use the packets received.
        '''

        if self.logger: self.logger.log(
            "Comm.responder",
            "Started responding"
        )
            
        empty = False
        while not self.stop_signal:

            '''
            Pop a packet from the queue and call the
            packet_handler with it as an argument.
            '''

            if empty: time.sleep(self.timeout)
            empty = False

            packet: Packet = None # type: ignore
            with self.packet_lock:

                try: packet = self.packets.pop(0)
                except:

                    empty = True
                    continue

            if packet: packet_handler(packet)

        if self.logger: self.logger.log(
            "Comm.responder",
            "Stopped responding."
        )

    
    def start(self, packet_handler: Callable [[Packet], None]) -> None:

        '''
        Starts the listeners and responder as child threads.
        '''

        self.stop_signal = False

        threading.Thread(
            target=self.responder, 
            args=(packet_handler, )
        ).start()

        threading.Thread(target=self.tcp_listener).start()
        threading.Thread(target=self.udp_listener).start()


    def stop(self) -> None:

        '''
        Signals the listeners and responder to stop.
        '''

        self.stop_signal = True


if __name__ == "__main__":

    '''
    Testing script.
    '''

    import time

    comm1: Comm = Comm(
        ip = "127.0.0.1",
        tcp_port = 4000,
        udp_port = 4001,
        logger = None,      # type: ignore
        sim = True
    )

    comm2: Comm = Comm(
        ip = "127.0.0.2",
        tcp_port = 4000,
        udp_port = 4001,
        logger = None,      # type: ignore
        sim = True
    )

    def packet_handler(packet: Packet):

        print("\n%s\n" % packet.payload.decode())

    comm1.start(packet_handler)
    comm2.start(packet_handler)

    input("\n[Press Enter to Start]\n")

    comm1.send(
        Packet(
            sender = (comm1.ip, 0),
            destination = (comm2.ip, comm1.tcp_port),
            type = "",
            payload = "Comm1 TCP packet to Comm2".encode()
        )
    )

    comm2.send(
        Packet(
            sender = (comm2.ip, 0),
            destination = (comm1.ip, comm2.tcp_port),
            type = "",
            payload = "Comm2 TCP packet to Comm1".encode()
        )
    )

    comm1.broadcast(
        Packet(
            sender = (comm1.ip, 0),
            destination = ("<broadcast>", comm1.udp_port),
            type = "",
            payload = "Comm1 UDP packet to ALL".encode()
        )
    )

    comm2.broadcast(
        Packet(
            sender = (comm2.ip, 0),
            destination = ("<broadcast>", comm2.udp_port),
            type = "",
            payload = "Comm2 UDP packet to ALL".encode()
        )
    )

    input("\n[Press Enter to Stop]\n")

    comm1.stop()
    comm2.stop()