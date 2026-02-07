import serial
from time import sleep
 
COM_PORT = 'COM7'  # 請自行修改序列埠名稱
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)


def watering(kind):
    if kind==1:
        arduComm='c'
    elif kind==2:
        arduComm='b'
    else:
        arduComm='t'
    ser.write(arduComm.encode())
    sleep(1.5)

def carMove():
    print('傳送開車指令')
    ser.write(b'g\n')
    carComm = True
    while carComm :
        arduBack = ser.read()
        print(arduBack)
        if arduBack == b'a':
            carComm = False

    return carComm

def serial_end():
    ser.close()
    
def test():
    watering(1)
    sleep(1)
    watering(2)
    sleep(1)