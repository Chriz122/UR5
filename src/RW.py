import cv2

global num
num = 1

def save_pic(img,string):
    num += 1
    name = "data/images/" + string + str(num) + ".png"
    cv2.imwrite(name, img)
    f = open("data/txts/num.txt",'w')
    f.write(str(num))
    f.close()
    
def f_close():
    global f
    f.close()

if __name__ == "__main__":
    f = open("data/txts/num.txt",'r')
    num =int(f.read())
    f.close()