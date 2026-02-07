import urx
import time
import cv2
import socket
from math import pi

t=0
C=10/1000

rob = urx.Robot("192.168.1.101")
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))
time.sleep(0.2)
OR=rob.getl()
pose=rob.getl()

temp=1
# Streaming loop

try:
    with open('D:\\a20230718a\\pose.txt','w') as s:
        while True:          
            cv2.namedWindow("Robot Hand Panel",0)
            key = cv2.waitKey(10)
            
            if key & 0xFF == ord('o'):
                s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                s.connect(("192.168.1.101",30001))
                
                f = open(r'open.txt')
                text = []
                for line in f:
                    text.append(line)
                strr = "".join(text)
                
                s.send(str.encode(strr))
                time.sleep(1)
                
                s.close()
                time.sleep(1)
                
                rob.set_tcp((0, 0, 0.1, 0, 0, 0))
                rob.set_payload(2, (0, 0, 0.1))
                time.sleep(0.2)
                
                print('open')
                
            if key & 0xFF == ord('c'):
                s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                s.connect(("192.168.1.101",30001))
                
                f = open(r'close.txt')
                text = []
                for line in f:
                    text.append(line)
                strr = "".join(text)
                
                s.send(str.encode(strr))
                time.sleep(1)
                
                s.close()
                time.sleep(1)
                
                rob.set_tcp((0, 0, 0.1, 0, 0, 0))
                rob.set_payload(2, (0, 0, 0.1))
                time.sleep(0.2)

                print('close')
                
            if key & 0xFF == ord('g'):
                aa=-0.24854226995698214-0.07
                bb=-0.5996794558010212-0.07
                rob.movel((aa,bb,0.39196252797369563,1.0372,2.5038,-2.5038),1,0.1)
                time.sleep(1)
                
                s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                s.connect(("192.168.1.101",30001))
                
                f = open(r'close.txt')
                text = []
                for line in f:
                    text.append(line)
                strr = "".join(text)
                
                s.send(str.encode(strr))
                time.sleep(1)
                
                s.close()
                time.sleep(1)
                
                rob.set_tcp((0, 0, 0.1, 0, 0, 0))
                rob.set_payload(2, (0, 0, 0.1))
                time.sleep(0.2)
                
                
                rob.movel_tool((0,0,-0.1,0,0,0),1,0.1)
                rob.movel((OR[0],OR[1],OR[2],1.0372,2.5038,-2.5038),1,0.1)
                time.sleep(1)
                
                s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                s.connect(("192.168.1.101",30001))
                
                f = open(r'open.txt')
                text = []
                for line in f:
                    text.append(line)
                strr = "".join(text)
                
                s.send(str.encode(strr))
                time.sleep(1)
                
                s.close()
                time.sleep(1)
                
                rob.set_tcp((0, 0, 0.1, 0, 0, 0))
                rob.set_payload(2, (0, 0, 0.1))
                time.sleep(0.2)
                
                print('done')
                
            if key & 0xFF == ord('x'):
                print(temp)
                temp+=1
                s.write('%f ' %pose[0])
                s.write('%f ' %pose[1])
                s.write('%f ' %pose[2])
                s.write('-90 0 135\n')
                
            if key & 0xFF == ord('h'):
                print(OR[0])
                print(OR[1])
                print(OR[2])
            
            if key & 0xFF == ord('j'):
                C=10/1000
                print(10)    
            if key & 0xFF == ord('k'):
                C=1/1000
                print(1)
            if key & 0xFF == ord('l'):
                C=0.1/1000
                print(0.1)
            
            if key & 0xFF == ord('z'):
                pose=rob.getl()
                print(pose[0])
                print(pose[1])
                print(pose[2])
            
            if key & 0xFF == ord('w'):
                pose[2]=pose[2]+C
                p=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],1.0372,2.5038,-2.5038),1,0.1)
            if key & 0xFF == ord('s'):
                pose[2]=pose[2]-C
                p=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],1.0372,2.5038,-2.5038),1,0.1)
            if key & 0xFF == ord('d'):
                pose[0]=pose[0]-C*pi/4
                pose[1]=pose[1]+C*pi/4
                p=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],1.0372,2.5038,-2.5038),1,0.1)
            if key & 0xFF == ord('a'):
                pose[0]=pose[0]+C*pi/4
                pose[1]=pose[1]-C*pi/4
                p=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],1.0372,2.5038,-2.5038),1,0.1)
            if key & 0xFF == ord('q'):
                pose[0]=pose[0]-C*pi/4
                pose[1]=pose[1]-C*pi/4
                p=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],1.0372,2.5038,-2.5038),1,0.1)
            if key & 0xFF == ord('e'):
                pose[0]=pose[0]+C*pi/4
                pose[1]=pose[1]+C*pi/4
                p=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],1.0372,2.5038,-2.5038),1,0.1)
            
            if key == 27:
                cv2.destroyAllWindows()
                break
                
                
finally:
    # rob.movel((OR[0],OR[1],OR[2],1.0372,2.5038,-2.5038),1,0.1)
    print('Finished')