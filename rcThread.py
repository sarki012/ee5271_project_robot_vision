# Erik Sarkinen
# EE 5271 Project


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
import streamThread
import robotMain
from threading import Event
import signal



data_char = 0

deviceUC1 = Device('FT7210GA')     #Bottom USB
deviceUC2 = Device('FT71VG2G')     #Top USB

client_sock = 0
server_sock = 0
def task1():
    global client_sock
    global server_sock

    cmd = 'sudo hciconfig hci0 piscan'
    os.system(cmd)
    j = 0
    wait = 0.001
    #F4:42:8F:10:5F:5F

    global connection
    connection = False

    server_sock=BluetoothSocket( RFCOMM )
    server_sock.bind(("",PORT_ANY))
    server_sock.listen(1)

    port = server_sock.getsockname()[1]
    uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
    #uuid = "00001101-0000-1000-8000-00805F9B34FB"
    advertise_service( server_sock, "raspberrypi",
                    service_id = uuid,
                    service_classes = [ uuid, SERIAL_PORT_CLASS ],
                    profiles = [ SERIAL_PORT_PROFILE ]  
                        )

    filedescriptors = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
  #  deviceUC1 = Device('FT7210GA')     #Bottom USB
    deviceUC1.baudrate = 115200
    deviceUC1.open()
   # deviceUC2 = Device('FT71VG2G')     #Top USB
    deviceUC2.baudrate = 115200
    deviceUC2.open()


    while True:
        if(connection == False):
            print("Waiting for connection on RFCOMM channel %d" % port)
            client_sock, client_info = server_sock.accept()
            connection = True
            print("Accepted connection from ", client_info)
        try:  
            data = client_sock.recv(5)      #was 50
            if len(data) == 0:
                break
            data_char = chr(data[0])
            if data_char == 'A':        #A for Auto Mode, toggleFlag = 1
                robotMain.thread_switch_event.set()
            if data_char == 'R':      #R for RC Mode, toggleFlag = 0
            #    print("Back to RC Mode!")
                robotMain.thread_switch_event.clear()
                robotMain.go_event.clear()
            if data_char == 'g' and robotMain.thread_switch_event.is_set():        #Go in auto mode
               # print("Go//////////////////////////////////////////")
                robotMain.go_event.set()        #Go in auto mode
            if data_char == 'S' and robotMain.thread_switch_event.is_set():        #Go in auto mode
                robotMain.go_event.clear()        #Go in auto mode
            if data_char == 'x':      #Stop
              #  print("Stop")
                robotMain.go_event.clear()        #Stop in auto mode
                deviceUC1.write('q')                #q for quit
                deviceUC2.write('q')                #q for quit
            if data_char == 'H':        #Up
                deviceUC2.write('H')
            elif data_char == 'u':        #Up
                deviceUC2.write('u')
            elif data_char == 'd':      #Down
                deviceUC2.write('d')
            elif data_char == '@':      #Stop boom
                deviceUC2.write('@')
            elif data_char == 'l':      #Left
                deviceUC1.write('l')
            elif data_char == 'r':      #Right
                deviceUC1.write('r')
            elif data_char == '$':      #Stop rotation
                deviceUC1.write('$')
            elif data_char == 'O':      #Out
                deviceUC2.write('O')
            elif data_char == 'I':      #In
                deviceUC2.write('I')
            elif data_char == '&':      #Stop Stick
                deviceUC2.write('&')
            elif data_char == 't':      #Tip down
                deviceUC2.write('t')
            elif data_char == 'p':      #Tip up
                deviceUC2.write('p')
            elif data_char == '^':      #Stop tip motor
                deviceUC2.write('^')
            elif data_char == 'n':      #CLaw open
                deviceUC1.write('n')
            elif data_char == 'c':      #Claw closed
                deviceUC1.write('c')
            elif data_char == '%':      #Stop claw motor
                deviceUC1.write('%')

        except KeyboardInterrupt:
            print("\nDisconnected")
            client_sock.close()
            server_sock.close()
            raise SystemExit 
#def task2():
 #   while True:
  #      if(connection != False):
   #         try:  
                ##########Data Relay From dsPIC UART to Bluetooth#########
               # adcValsUc1 = deviceUC1.read(10)
              #  adcValsUc1 = rcThread.deviceUC1.read(20)     #Was 20
               # print(adcValsUc1)
                #client_sock.send(adcValsUc1)
                #time.sleep(0.1)
                #adcValsUc2 = deviceUC2.read(25)     #Was 25
                #print(adcValsUc2)
                #client_sock.send(adcValsUc2)
                #time.sleep(0.1)
                #time.sleep(0.1)
    #        except KeyboardInterrupt:
      #          print("\nDisconnected")
     #           raise SystemExit 

#def task3():
 #   while True:
  #      if(connection != False):
   #         try:  
    #            ##########Data Relay From dsPIC UART to Bluetooth#########
     #           global adcValsUc1
      #          adcValsUc1 = deviceUC1.read(10)
       #         print(adcValsUc1)
        #        #time.sleep(0.1)
         #   except KeyboardInterrupt:
          #      print("\nDisconnected")
           #     raise SystemExit 