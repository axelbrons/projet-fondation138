
import machine
import network
import time
from machine import Pin, PWM
from umqtt.simple import MQTTClient
import socket

# Connections for drive Motors
PWM_A = machine.Pin(5)  # D1
PWM_B = machine.Pin(4)  # D2
DIR_A = machine.Pin(0)  # D3
DIR_B = machine.Pin(2)  # D4
buzPin = machine.Pin(14, machine.Pin.OUT)  # D5
ledPin = machine.Pin(15, machine.Pin.OUT)  # D8
wifiLedPin = machine.Pin(16, machine.Pin.OUT)  # D0

# Initialize the pins
DIR_A.init(Pin.OUT)
DIR_B.init(Pin.OUT)
PWM_A = PWM(PWM_A, freq=1000)
PWM_B = PWM(PWM_B, freq=1000)

# Set initial values
DIR_A.value(0)
DIR_B.value(0)
PWM_A.duty(0)
PWM_B.duty(0)
buzPin.value(0)
ledPin.value(0)
wifiLedPin.value(1)

SPEED = 1023
speed_Coeff = 3
command = ""

# WiFi Configuration
sta_ssid = ""
sta_password = ""


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(sta_ssid, sta_password)
    start_time = time.time()

    while not wlan.isconnected() and time.time() - start_time < 10:
        print('.', end='')
        time.sleep(0.5)

    if wlan.isconnected():
        print("\n*WiFi-STA-Mode*")
        print("IP: ", wlan.ifconfig()[0])
        wifiLedPin.value(0)
        time.sleep(3)
    else:
        wlan.active(False)
        ap = network.WLAN(network.AP_IF)
        ap.active(True)
        ap.config(essid='wificar-{}'.format(machine.unique_id()))
        print("\n*WiFi-AP-Mode*")
        print("AP IP address: ", ap.ifconfig()[0])
        wifiLedPin.value(1)
        time.sleep(3)


def handle_client(client):
    global command, SPEED
    request = client.recv(1024)
    request = str(request)
    command_start = request.find('/?State=')
    command_end = request.find(' ', command_start)

    if command_start != -1:
        command = request[command_start + 8:command_end]
        print("Command:", command)

    if command == "F":
        Forward()
    elif command == "B":
        Backward()
    elif command == "R":
        TurnRight()
    elif command == "L":
        TurnLeft()
    elif command == "G":
        ForwardLeft()
    elif command == "H":
        BackwardLeft()
    elif command == "I":
        ForwardRight()
    elif command == "J":
        BackwardRight()
    elif command == "S":
        Stop()
    elif command == "V":
        BeepHorn()
    elif command == "W":
        TurnLightOn()
    elif command == "w":
        TurnLightOff()
    elif command.isdigit() and 0 <= int(command) <= 9:
        SPEED = 330 + int(command) * 70
    elif command == "q":
        SPEED = 1023

    response = "HTTP/1.1 200 OK\n\n"
    client.send(response)
    client.close()


def Forward():
    DIR_A.value(1)
    DIR_B.value(1)
    PWM_A.duty(SPEED)
    PWM_B.duty(SPEED)


def Backward():
    DIR_A.value(0)
    DIR_B.value(0)
    PWM_A.duty(SPEED)
    PWM_B.duty(SPEED)


def TurnRight():
    DIR_A.value(0)
    DIR_B.value(1)
    PWM_A.duty(SPEED)
    PWM_B.duty(SPEED)


def TurnLeft():
    DIR_A.value(1)
    DIR_B.value(0)
    PWM_A.duty(SPEED)
    PWM_B.duty(SPEED)


def ForwardLeft():
    DIR_A.value(1)
    DIR_B.value(1)
    PWM_A.duty(SPEED)
    PWM_B.duty(SPEED // speed_Coeff)


def BackwardLeft():
    DIR_A.value(0)
    DIR_B.value(0)
    PWM_A.duty(SPEED)
    PWM_B.duty(SPEED // speed_Coeff)


def ForwardRight():
    DIR_A.value(1)
    DIR_B.value(1)
    PWM_A.duty(SPEED // speed_Coeff)
    PWM_B.duty(SPEED)


def BackwardRight():
    DIR_A.value(0)
    DIR_B.value(0)
    PWM_A.duty(SPEED // speed_Coeff)
    PWM_B.duty(SPEED)


def Stop():
    DIR_A.value(0)
    DIR_B.value(0)
    PWM_A.duty(0)
    PWM_B.duty(0)


def BeepHorn():
    buzPin.value(1)
    time.sleep(0.15)
    buzPin.value(0)
    time.sleep(0.08)


def TurnLightOn():
    ledPin.value(1)


def TurnLightOff():
    ledPin.value(0)


def start_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 80))
    s.listen(5)

    while True:
        conn, addr = s.accept()
        print('Got a connection from %s' % str(addr))
        handle_client(conn)


connect_wifi()
start_server()
