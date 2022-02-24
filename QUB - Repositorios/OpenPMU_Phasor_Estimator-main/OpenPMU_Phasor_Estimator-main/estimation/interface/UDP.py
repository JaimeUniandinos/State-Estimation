"""
OpenPMU - Phasor Estimator
Copyright (C) 2021  www.OpenPMU.org

Licensed under GPLv3.  See __init__.py
"""

import sys

import socket
import estimation.interface.phasor as phasor

__author__ = 'Xiaodong'


class Receiver():
    """
    UDP interface to receive phasor estimation results
    """

    def __init__(self, ip , port):
        self.socketIn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socketIn.bind((ip, port))

    def receive(self, ):
        """
        Receive phasor estimation results.

        :return: python dict of the result
        """
        xml, __address = self.socketIn.recvfrom(8192)
        return phasor.fromXML(xml)


class Sender():
    """
    UDP interface to send phasor estimation results
    """

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socketOut = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, resultDict):
        """
        send phasor estimation results

        :param resultDict: python dict of the result
        :return:
        """
        xml = phasor.toXML(resultDict)

        self.socketOut.sendto(xml, (self.ip, self.port))
