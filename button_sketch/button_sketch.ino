const int buttonPin = 2;

int lastState = HIGH;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  Serial.begin(9600);
}

void loop() {
  int currentState = digitalRead(buttonPin);

  // Button just pressed
  if (lastState == HIGH && currentState == LOW) {
    Serial.println("PRESS");
    delay(30);  // debounce
  }

  lastState = currentState;
}