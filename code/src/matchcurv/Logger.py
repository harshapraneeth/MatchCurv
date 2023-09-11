import os
import shutil
import logging
from datetime import datetime

class Logger:

    '''
    Wrapper for logging library
    '''

    def __init__(
        self, 
        location: str = "logs/",
        prefix: str = "log",
        stdout: bool = False
    ):
        
        '''
        Here we create the log file using,
        location - the location of the logs folder (should always end with "/").
        prefix - the name of the log file would be "prefix-current_time.log".
        stdout - True if we want to see everything being logged in the terminal.
        '''

        self.stdout = stdout

        '''
        Remove this if you don't want to empty the logs directory.
        Only empty the directory outside the simulation.
        '''

        '''
        if os.path.exists(location):
            shutil.rmtree(location)
        '''

        '''
        Create the directory if it doesn't exist.
        And create a log file "location/prefix-year-month-day-hour-minute-second.log".
        '''

        if not os.path.exists(location): 
            os.makedirs(location)

        name: str = prefix + "-" + datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )

        logging.basicConfig(
            filename = location + name + ".log",
            format = "(%(asctime)s) %(message)s"
        )

        self.logger: logging.Logger = logging.getLogger(prefix)
        self.logger.setLevel(logging.DEBUG)


    def log(
        self, 
        source: str, 
        message: str, 
        *args
    ) -> None:

        '''
        This method writes a message to the log file.
        Every log message is formatted as, "(time) ip | source> message".
        args - used for message formatting.
        '''

        self.logger.info("%s> %s" % (source, message % args))

        if not self.stdout: return

        now: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        print("(%s) %s> %s" % (now, source, message % args))


if __name__ == "__main__":

    '''
    Testing script
    '''

    logger = Logger(
        location = "../../logs/",
        prefix = "test",
        stdout = True
    )

    logger.log(
        "Main",
        "Hello int: %d, float %.3f, string: %s",
        1, 2, "Test"
    )