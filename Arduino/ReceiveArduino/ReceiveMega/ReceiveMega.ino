#include <SoftwareSerial.h>
SoftwareSerial BTSerial(50, 51); // RX | TX <------------------- Change according to your pin number. 

int sendcmd(int); 

void setup() 
{
    Serial.begin(9600);
    Serial.println("Arduino with HC-06 is ready");

    // HC-06 default baud rate is 9600
    BTSerial.begin(9600);  
    Serial.println("BTserial started at 9600");
}

void loop()
{
  //BTSerial.println("Test"); 
  bool control = true; 
  while (control) {
    if (BTSerial.available()) {
      char c = BTSerial.read(); // Get keypress from Python. 
      Serial.write(c); 
      int d = c - '0'; 
      switch (d) {
        case 0: Serial.write("-Stop\n"); break; 
        case 1: Serial.write("-Front\n"); break; // Servo functions. Blah blah...
        case 2: Serial.write("-Back\n"); break; 
        case 3: Serial.write("-Left\n"); break; 
        case 4: Serial.write("-Right\n"); break; 
        case 5: Serial.write("-Anti-CW\n"); break; 
        case 6: Serial.write("-Clockwise\n"); break; 
        case 7: control=false; break; 
        default: break; 
      }
    }
  }

  // Auto drive start. 
  Serial.write("-Auto-"); 
  int digit = sendcmd(1); 
  Serial.write(digit); 
  delay(100); 
  switch (digit) {
    case 0: break; 
    case 1: break; 
    case 2: break; 
    case 3: break; 
    case 4: break; 
    case 5: break; 
    case 6: break; 
    case 7: break; 
    case 8: break; 
    case 9: break; 
  }
  // Blah blah...
  Serial.write("\nEnd"); 
}

int sendcmd(int mode) { // 求辨识数字并等待返回 Request for digit recognization and wait for return. 
                        // Mode 0: Highest speed. 
                        // Mode 1: Highest accuracy. 
  delay(500); 
  BTSerial.println('1'); // Send data from Arduino Serial to Python. 
  Serial.write("Sent\n"); 
  int result; 
  while (1) {
    if (BTSerial.available()) {
      result = BTSerial.read(); // Get data from Python. 
      break; 
    }
  } 

  return result; 
}
