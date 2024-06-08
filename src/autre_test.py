import network
import machine
import time
import esp
from machine import Pin, PWM

# connections for drive Motors
PWM_A = machine.Pin('D1')
PWM_B = machine.Pin('D2')
DIR_A = machine.Pin('D3')
DIR_B = machine.Pin('D4')
buzPin = machine.Pin('D5', machine.Pin.OUT)  # set digital pin D5 as buzzer pin (use active buzzer)
ledPin = machine.Pin('D8', machine.Pin.OUT)  # set digital pin D8 as LED pin (use super bright LED)
wifiLedPin = machine.Pin('D0', machine.Pin.OUT)  # set digital pin D0 as indication, the LED turn on if NodeMCU connected to WiFi as STA mode

command = ''  # String to store app command state
SPEED = 1023  # 330 - 1023
speed_Coeff = 3

sta_ssid = ''  # set Wifi networks you want to connect to
sta_password = ''  # set password for Wifi networks

def setup():
    print('*WiFi Robot Remote Control Mode*')
    print('--------------------------------------')
    buzPin.value(0)
    ledPin.value(0)
    wifiLedPin.value(1)  # Set all the motor control pins to outputs
    DIR_A.value(0)
    DIR_B.value(0)
    PWM_A.value(0)
    PWM_B.value(0)

    # set NodeMCU Wifi hostname based on chip mac address
    chip_id = hex(esp.flash_id())[2:]
    hostname = 'wificar-' + chip_id[-4:]
    print('Hostname:', hostname)

    # first, set NodeMCU as STA mode to connect with a Wifi network
    sta_if = network.WLAN(network.STA_IF)
    sta_if.active(True)
    sta_if.connect(sta_ssid, sta_password)
    print('Connecting to:', sta_ssid)
    print('Password:', sta_password)

    # try to connect with Wifi network about 10 seconds
    start_time = time.time()
    while not sta_if.isconnected() and time.time() - start_time <= 10:
        time.sleep(0.5)
        print('.', end='')

    # if failed to connect with Wifi network set NodeMCU as AP mode
    if sta_if.isconnected():
        print('*WiFi-STA-Mode*')
        print('IP:', sta_if.ifconfig()[0])
        wifiLedPin.value(0)  # Wifi LED on when connected to Wifi as STA mode
        time.sleep(3)
    else:
        ap_if = network.WLAN(network.AP_IF)
        ap_if.active(True)
        ap_if.config(essid=hostname)
        print('WiFi failed connected to', sta_ssid)
        print('*WiFi-AP-Mode*')
        print('AP IP address:', ap_if.ifconfig()[0])
        wifiLedPin.value(1)  # Wifi LED off when status as AP mode
        time.sleep(3)

setup()

def loop():
    ArduinoOTA.handle()  # listen for update OTA request from clients
    server.handle_client()  # listen for HTTP requests from clients
    command = server.arg("State")  # check HTPP request, if has arguments "State" then saved the value
    if command == "F":
        forward()  # check string then call a function or set a value
    elif command == "B":
        backward()
    elif command == "R":
        turn_right()
    elif command == "L":
        turn_left()
    elif command == "G":
        forward_left()
    elif command == "H":
        backward_left()
    elif command == "I":
        forward_right()
    elif command == "J":
        backward_right()
    elif command == "S":
        stop()
    elif command == "V":
        beep_horn()
    elif command == "W":
        turn_light_on()
    elif command == "w":
        turn_light_off()
    elif command == "0":
        SPEED = 330
    elif command == "1":
        SPEED = 400
    elif command == "2":
        SPEED = 470
    elif command == "3":
        SPEED = 540
    elif command == "4":
        SPEED = 610
    elif command == "5":
        SPEED = 680
    elif command == "6":
        SPEED = 750
    elif command == "7":
        SPEED = 820
    elif command == "8":
        SPEED = 890
    elif command == "9":
        SPEED = 960
    elif command == "q":
        SPEED = 1023

def HTTP_handle_root():
    server.send(200, "text/html", "")  # Send HTTP status 200 (Ok) and send some text to the browser/client
    if server.has_arg("State"):
        print(server.arg("State"))

def handle_not_found():
    server.send(404, "text/plain", "404: Not found")  # Send HTTP status 404 (Not Found) when there's no handler for the URI in the request

def forward():
    digitalWrite(DIR_A, True)
    digitalWrite(DIR_B, True)
    analogWrite(PWM_A, SPEED)
    analogWrite(PWM_B, SPEED)

def backward():
    digitalWrite(DIR_A, False)
    digitalWrite(DIR_B, False)
    analogWrite(PWM_A, SPEED)
    analogWrite(PWM_B, SPEED)

def turn_right():
    digitalWrite(DIR_A, False)
    digitalWrite(DIR_B, True)
    analogWrite(PWM_A, SPEED)
    analogWrite(PWM_B, SPEED)

def turn_left():
    digitalWrite(DIR_A, True)
    digitalWrite(DIR_B, False)
    analogWrite(PWM_A, SPEED)
    analogWrite(PWM_B, SPEED)

def forward_left():
    digitalWrite(DIR_A, True)
    digitalWrite(DIR_B, True)
    analogWrite(PWM_A, SPEED)
    analogWrite(PWM_B, SPEED / speed_Coeff)

def backward_left():
    digitalWrite(DIR_A, False)
    digitalWrite(DIR_B, False)
    analogWrite(PWM_A, SPEED)
    analogWrite(PWM_B, SPEED / speed_Coeff)

def forward_right():
    digitalWrite(DIR_A, True)
    digitalWrite(DIR_B, True)
    analogWrite(PWM_A, SPEED / speed_Coeff)
    analogWrite(PWM_B, SPEED)

def backward_right():
    digitalWrite(DIR_A, False)
    digitalWrite(DIR_B, False)
    analogWrite(PWM_A, SPEED / speed_Coeff)
    analogWrite(PWM_B, SPEED)

def stop():
    digitalWrite(DIR_A, False)
    digitalWrite(DIR_B, False)
    analogWrite(PWM_A, 0)
    analogWrite(PWM_B, 0)

def beep_horn():
    digitalWrite(buzPin, True)
    time.sleep(0.15)
    digitalWrite(buzPin, False)
    time.sleep(0.08)

def turn_light_on():
    digitalWrite(ledPin, True)

def turn_light_off():
    digitalWrite(ledPin, False)

