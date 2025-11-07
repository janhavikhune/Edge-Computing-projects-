import Jetson.GPIO as GPIO
import time

class PIRSensor:
    def __init__(self, pin=23):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)

    def detect_motion(self):
        """Return True when motion is detected."""
        return GPIO.input(self.pin)

    def cleanup(self):
        GPIO.cleanup()


if __name__ == "__main__":
    pir = PIRSensor()
    print("🔋 PIR sensor ready — move in front of it to test.")
    try:
        while True:
            if pir.detect_motion():
                print("👀 Motion detected!")
                time.sleep(1)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pir.cleanup()
        print("⚙️ Clean exit.")
