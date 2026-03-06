import urx
import time
import socket
from math import radians

UR5 = {
    "IP": "192.168.1.101",
    "PORT": 30001
}
acc = 0.1
vel = 0.3
rob = urx.Robot(UR5["IP"])

rob.set_tcp((0, 0, 0.1, 0, 0, 0))

rob.set_payload(2, (0, 0, 0.1))
time.sleep(0.2)

def OPEN():
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((UR5["IP"],UR5["PORT"]))
    
    f = open(r'src/open.txt')
    text = []
    for line in f:
        text.append(line)
    strr = "".join(text)
    
    s.send(str.encode(strr))
    time.sleep(0.1)
    
    s.close()
    time.sleep(0.1)
    
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))
    
    print('open')
    
def CLOSE(width):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((UR5["IP"],UR5["PORT"]))
    
    f = open(r'src/close.txt')
    text = []
    for line in f:
        if "rq_set_pos_norm(100, " in line: # 8.5cm = 0.085 m
            pos = abs(round((width)/0.00085) - 100)
            line = "    rq_set_pos_norm(" + str(pos) + ', "1")\n'
            text.append(line)
        else:    
            text.append(line)
    
    strr = "".join(text)
    
    s.send(str.encode(strr))
    time.sleep(0.5)
    
    s.close()
    time.sleep(0.5)
    
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))

    print('close')

if __name__ == "__main__":
    close_length = 0.05

    '''夾取測試'''
    CLOSE(close_length)
    # 移動到目標位置
    time.sleep(1)
    rob.movej((radians(15.17),
                radians(-59.64),
                radians(-124.58),             # 負的向下
                radians(-84.99),
                radians(88.90),
                radians(105.10)),acc,vel)
    CLOSE(close_length*0.7)

    # 預備姿勢
    time.sleep(1)
    rob.movej((radians(-26.15),
                radians(-71.91),
                radians(-106.04),             # 負的向下
                radians(-91.31),
                radians(89.45),
                radians(63.88)),acc,vel)  #校正角度
    CLOSE(close_length*0.7)
    

    '''放置測試'''
    # 移動到目標位置
    time.sleep(1)
    rob.movej((radians(15.17),
                radians(-59.64),
                radians(-124.58),             # 負的向下
                radians(-84.99),
                radians(88.90),
                radians(105.10)),acc,vel)
    CLOSE(close_length*0.7)

    for _ in range(2):
        CLOSE(close_length*1.5)
        CLOSE(close_length*0.7)
    CLOSE(close_length)

    # 預備姿勢
    time.sleep(1)
    rob.movej((radians(-26.15),
                radians(-71.91),
                radians(-106.04),             # 負的向下
                radians(-91.31),
                radians(89.45),
                radians(63.88)),acc,vel)  #校正角度
    CLOSE(close_length)

    rob.close()