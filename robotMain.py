# Erik Sarkinen

from pylibftdi import Device
import os
import glob
import time
from bluetooth import *
import socket
import bluetooth
import cv2
import numpy as np
import tty, sys, termios
import threading
import rcThread
import streamThread
import feedbackThread
import visualServoing
from threading import Event
import queue

thread_switch_event = Event()
go_event = Event()

queue = queue.Queue()


if __name__ == "__main__":

    # creating threads
    t1 = threading.Thread(target=rcThread.task1, name='t1')
    t2 = threading.Thread(target=feedbackThread.task2, name='t2') 
    t3 = threading.Thread(target=visualServoing.task3, name='t3') 
    t4 = threading.Thread(target=streamThread.task4, name='t4') 
 
    # starting threads
    t1.start()
    time.sleep(1)
    t2.start()
    #time.sleep(1)
    t3.start()
    t4.start()
 
    # wait until all threads finish
    t1.join()
    t2.join()
    t3.join()
    t4.join()