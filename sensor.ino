// Define the pin for the PIR sensor
int pirPin = 4;
int pin=5; // Change this to the pin your PIR sensor is connected to

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  // Set the PIR pin as input
  pinMode(pirPin, INPUT);
  pinMode(pin, OUTPUT);
}

void loop() {
  // Read the value from the PIR sensor
  int pirValue = digitalRead(pirPin);

  // Check if motion is detected
  if (pirValue == HIGH) {
    Serial.println("1");
    digitalWrite(pin, HIGH);
     // Sending '1' to indicate motion detected
  } else {
    digitalWrite(pin, LOW);
    Serial.println("0"); // Sending '0' to indicate no motion detected
  }

  // Delay for a short period before reading again
  delay(3000);
}
