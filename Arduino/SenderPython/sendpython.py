import serial 
import time 

def main(): 
    port = "COM5" #may change. 
    bluetooth = serial.Serial(port, 9600)
    print("Connected")
    bluetooth.flushInput() #kickstart. 
    i = 2
    bluetooth.write(str.encode(str(i)))
    
    print("write"); 
    
    while 1: 
        leng = bluetooth.inWaiting()
        if (leng > 0): 
            input_data = bluetooth.readline()
            print(str(leng)+":"+str(input_data.decode()))
            break 
    
    time.sleep(0.1)
    bluetooth.close()
    
if __name__ == "__main__": 
    main() 