import cv2

f = open("D:\\20210514\\pic2\\num.txt",'r')
num =int(f.read())
f.close()

def save_pic(img,string):
    global num,f
    num += 1
    name = "D:\\20210514\\pic2\\" + string + str(num) + ".bmp"
    cv2.imwrite(name, img)
    f = open("D:\\20210514\\pic2\\num.txt",'w')
    f.write(str(num))
    f.close()
    
def f_close():
    global f
    f.close()
