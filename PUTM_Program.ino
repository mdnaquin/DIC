#include <string.h>
#define ENCODER_OPTIMIZE_INTERRUPTS
#include <Encoder.h>


Encoder myEnc(2,3);
const int pressurePin = A7;
const int LightPin = 4;
int hold = 1;
double lastTime = 0;


void setup() {
  Serial.begin(2000000);
  pinMode(LightPin, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
}

long oldPosition  = -999;

void loop() {
  const double readPressureRaw = analogRead(pressurePin);
  const double calculatedPressure = ((readPressureRaw/1024*150*(1/.854))-16.75);
    if(calculatedPressure > 2){
      digitalWrite(LightPin, HIGH);
      digitalWrite(LED_BUILTIN, HIGH);
        if(hold ==1){
          lastTime = millis(); 
        }
        hold = 0;
    }else{
      digitalWrite(LightPin, LOW);
      digitalWrite(LED_BUILTIN, LOW);
      hold = 1;
    }
    if(calculatedPressure>2){
      Serial.print((millis() - lastTime)/1000);
    }else{
      Serial.print(0);
    }
    Serial.print(",");

    Serial.print(calculatedPressure);
    
    Serial.print(",");
    long newPosition = myEnc.read();
  if (newPosition != oldPosition) {
    oldPosition = newPosition;
    Serial.println(newPosition);
  }else{
    Serial.println(oldPosition);
  }
}
