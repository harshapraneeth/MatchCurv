class Packet:

    '''
    Only a packet object is transmitted between devices.
    The sender and destination are tuples (ip, port).
    The packet type indicates which function is invoked by the receiving device.
    The payload is used by the invoked function.
    '''

    def __init__(
        self,
        sender: tuple [str, int], 
        destination: tuple [str, int],
        type: str,
        payload: bytes
    ) -> None:
        
        self.sender = sender
        self.destination = destination
        self.type = type
        self.payload = payload

