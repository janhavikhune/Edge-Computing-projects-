import Jetson.GPIO as GPIO
import time

# Pin configuration
LED_PIN = 18
BUZZER_PIN = 15
SERVO_PIN = 33

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Setup pins
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Initialize off state
GPIO.output(LED_PIN, GPIO.LOW)
GPIO.output(BUZZER_PIN, GPIO.LOW)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# Alert functions
def led_alert(duration=1):
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(LED_PIN, GPIO.LOW)

def buzzer_alert(duration=0.5):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def servo_alert(angle=90):
    duty = 2 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)

def cleanup():
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    servo.stop()
    GPIO.cleanup()
