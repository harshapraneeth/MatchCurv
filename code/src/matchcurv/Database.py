'''import firebase_admin
from firebase_admin import db
from firebase_admin import credentials'''
from typing import Any

from Logger import *


class Database:

    '''
    This class handles all communication with the cloud database (Firebase).
    The cloud database is used for communicating with the dashboard,
    which we use to monitor the training.
    '''

    def __init__(
        self, 
        logger: Logger, 
        certificate: str,
        url: str, 
        id: str
    ) -> None:
        
        '''
        The certificate is used to authorize the access.
        The url is the url of the database.
        The id is the device id used in the path to store the related values.
        For example, /id/train_acc: [0,..,1].
        '''

        self.logger = logger
        self.certificate = certificate
        self.url = url
        self.id = id

        '''self.app: firebase_admin.App
        self.ref: db.Reference'''

        self.connected = False


    def connect(self) -> None:

        '''
        Here we establish the connection to the database.
        '''

        '''self.app = firebase_admin.initialize_app(
            credentials.Certificate(self.certificate), 
            {'databaseURL': self.url}
        )

        if len(self.id) > 0: 
            
            db.reference("entries").update({
                self.id: self.id
            })

        self.ref = db.reference(self.id)'''
        self.connected = True

        if self.logger: self.logger.log(
            "Device.connect",
            "Device connected to the database."
        )


    def reset(self) -> None:

        '''
        We should reset the values at the /id/ path.
        '''

        '''try:

            self.ref.child("round").set(0)
            self.ref.child("shares").set(0)
            self.ref.child("local_acc").set({0: 0})
            self.ref.child("local_loss").set({0: 0})
            self.ref.child("global_acc").set({0: 0})
            self.ref.child("global_loss").set({0: 0})

        except Exception as e:

            if self.logger: self.logger.log(
                "Database.reset",
                "Failed with exception [%s].",
                str(e)
            )'''


    def set(self, path: str, data: Any) -> None:

        '''
        Set method sets a values at certain path. 
        If a value already exists, it is overwritten.
        '''

        '''try:

            self.ref.child(path).set(data)

            if self.logger: self.logger.log(
                "Device.set",
                "Set %s in the database.",
                path
            )
                
        except Exception as e:

            if self.logger: self.logger.log(
                "Database.set",
                "Failed with exception [%s].",
                str(e)
            )'''


    def update(self, path: str, data: Any) -> None:

        '''
        Instead of set, we can use update to update an existing value.
        This is handy to update a child value without resetting the parent.
        '''

        '''try:

            self.ref.child(path).update(data)

            if self.logger: self.logger.log(
                "Device.update",
                "Updated %s in the database.",
                path
            )
                
        except Exception as e:

            if self.logger: self.logger.log(
                "Database.update",
                "Failed with exception [%s].",
                str(e)
            )'''


    def disconnect(self) -> None:

        '''
        Severs the connection with the database.
        '''

        '''try:

            firebase_admin.delete_app(self.app)

            if self.logger: self.logger.log(
                "Device.disconnected",
                "Disconnected from the database."
            )
                
        except Exception as e:

            if self.logger: self.logger.log(
                "Database.disconnect",
                "Failed with exception [%s].",
                str(e)
            )'''


if __name__ == "__main__":

    '''database = Database(
        None,               # type: ignore
        certificate = "!!!REDACTED!!!",
        url = "!!!REDACTED!!!",
        id = "device0"
    )

    database.connect()

    database.reset()

    database.set(
        "attributes", {"mu": "0.1", "lr": "0.01"}
    )

    database.update(
        "attributes", {"mu": "0.25", "cb": "0.1"}
    )

    database.disconnect()'''