# import socket
import time
import threading
import function_intelFORyolo as f_intel
import function_arm as f_arm
# import urx
import carMainCode as car

# rob = urx.Robot("192.168.1.101")

#HOST = "192.168.1.101"    # The remote host
#PORT = 30002              # The same port as used by the server
print ("Starting Progsram")
carComm=False
count=0
count_intel=0
acc = 0.5
vel = 0.3

def close_all():
    f_intel.intel_end()
    car.serial_end()
    f_arm.arm_close()
    
def imageObjDect():
    print('image')
    global count_intel
    while count_intel < 3:
        FF = f_intel.intel()
        if FF:
            count_intel+=1
        elif FF==None:
            break
        else:
            count_intel=3
        time.sleep(1)\
        
t = threading.Thread(target = imageObjDect)
t.start()
while(count < 30000):
    
    if not carComm :
        
        #預備點+左上
        print("##Start##")
        f_arm.arm_movej((0,-3.1416/2,0,-3.1416/2,-3.1416/4,0),acc,vel+0.2)
        time.sleep(0.1)
        print("##STEP1##")
        f_arm.arm_movej((0.098262,-1.122072,-1.281421,-0.737053,1.461015,-0.012566),acc,vel+0.2)
        time.sleep(0.1)
        
        #左下下
        f_arm.arm_movel((0.3,-0.1,0.16,2.5173,-2.5172,1.1739),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm_movel_tool((0,-0.05,0.05,0,0,0),acc,vel)
            #car.watering(2)
        #car.test()
        #左下
        f_arm.arm_movel((0.3,-0.1,0.3,2.5057,-2.5057,1.9227),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
            
        #car.test()
        
        #左中1
        f_arm.arm_movel((0.3,-0.1,0.400,2.4184,-2.4184,2.4184),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
        #car.test()
        
        #左中2
        f_arm.arm_movel((0.3,-0.1,0.520,2.4184,-2.4184,2.4184),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
        #car.test()
        
        #左上
        f_arm.arm_movel((0.3,-0.1,0.640,2.4184,-2.4184,2.4184),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
        #car.test()
        
        
        #預備點+右上
        f_arm.arm_movej((0,-3.1416/2,0,-3.1416/2,-3.1416/4,0),acc,vel+0.2)
        time.sleep(0.1)
        f_arm.arm_movej((-0.084474,-2.016204,1.288751,-2.412918,-1.497492,-0.009948),acc,vel+0.2)
        time.sleep(0.1)
        
        #右下下
        f_arm.arm_movel((-0.3,-0.1,0.16,2.5180,2.5170,-1.1738),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm_movel_tool((0,-0.05,0.05,0,0,0),acc,vel)
            #car.watering(2)
        #car.test()
        
        #右下
        f_arm.arm_movel((-0.3,-0.1,0.3,2.5057,2.5057,-1.9227),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
        #car.test()
        
        #右中1
        f_arm.arm_movel((-0.3,-0.1,0.400,2.4184,2.4184,-2.4184),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
        #car.test()
        
        #右中2
        f_arm.arm_movel((-0.3,-0.1,0.520,2.4184,2.4184,-2.4184),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
        #car.test()
        
        #右上
        f_arm.arm_movel((-0.3,-0.1,0.640,2.4184,2.4184,-2.4184),acc,vel)
        X,Y,Z,kind=f_intel.getXYZ()
        if len(kind) != 0:
            pose=f_arm.arm_getl()
            f_arm.arm(X,Y,Z,kind,pose,acc,vel)
        #car.test()
        
        
        #預備點
        f_arm.arm_movej((0,-3.1416/2,0,-3.1416/2,-3.1416/4,0),acc,vel+0.2)
        
        # data = s.recv(1024)
        # s.close()
        # print ("Received", repr(data))
        print ("Program finish")
        carComm=True
        
    else:
        f_arm.arm_close()
        carComm=car.carMove()
        #carComm=False
        count+=1
        count_intel=0
        time.sleep(0.5)
        f_arm.restart()
    
f_intel.intel_end()
car.serial_end()
f_arm.arm_close()
t.join()
print("結束噴藥")
