/*
 * ============================================================
 *  InMoov i2 Robotic Hand — EEG BCI Serial Control
 *  Hardware: Adafruit PCA9685 PWM Servo Driver (I2C)
 * ============================================================
 *  Receives string commands from Python (realtime.py):
 *    "mid\n"  →  Open hand   (all fingers to SERVOMID)
 *    "min\n"  →  Close hand  (all fingers to SERVOMIN)
 *
 *  Responds with:
 *    "OPEN\n"   after opening
 *    "CLOSED\n" after closing
 *    "Ready!\n" on startup
 *
 *  Finger channel assignments (PCA9685):
 *    0 = Thumb
 *    1 = Index
 *    2 = Middle
 *    3 = Ring
 *    4 = Wrist/Pinky
 *
 *  Libraries required:
 *    Adafruit PWM Servo Driver Library
 *    (Install via Arduino Library Manager)
 * ============================================================
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define THUMB   0
#define INDEX   1
#define MIDDLE  2
#define RING    3
#define WRIST   4

// PWM pulse counts for your servos (tune these to your mechanical limits)
#define SERVOMIN  150   // Close position
#define SERVOMID  375   // Open  position
#define SERVOMAX  600   // (reserved for future individual finger control)

#define N_FINGERS 5
#define BAUD_RATE 9600
#define MOVE_DELAY 100  // ms between fingers (prevents power surge)

// ── Helper functions ──────────────────────────────────────────

void openAll() {
  for (int i = 0; i < N_FINGERS; i++) {
    pwm.setPWM(i, 0, SERVOMID);
    delay(MOVE_DELAY);
  }
  Serial.println("OPEN");
}

void closeAll() {
  for (int i = 0; i < N_FINGERS; i++) {
    pwm.setPWM(i, 0, SERVOMIN);
    delay(MOVE_DELAY);
  }
  Serial.println("CLOSED");
}

// ── Setup ─────────────────────────────────────────────────────

void setup() {
  Serial.begin(BAUD_RATE);
  pwm.begin();
  pwm.setPWMFreq(50);   // Standard servo frequency: 50 Hz
  delay(10);
  openAll();            // Start in open position
  Serial.println("Ready!");
}

// ── Main loop ─────────────────────────────────────────────────

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "mid") {
      openAll();
    } else if (cmd == "min") {
      closeAll();
    }
    // ignore unrecognised commands
  }
}
