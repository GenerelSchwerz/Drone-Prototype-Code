import RPi.GPIO as GPIO
import time

# set GPIO Pins
GPIO_TRIGGER = 18
GPIO_ECHO = 24


class DistanceSensor:

    def __init__(self, trigger_pin: int, echo_pin: int):
        # GPIO Mode (BOARD / BCM)
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(GPIO_ECHO, GPIO.IN)
        self.trigger = trigger_pin
        self.echo = echo_pin
        self.last_update = time.time()

    # TODO.
    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def get_distance(self):
        GPIO.output(GPIO_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(GPIO_TRIGGER, False)

        start_time = time.time()
        while GPIO.input(GPIO_ECHO) == 0:
            start_time = time.time()


        self.last_update = time.time()
        while GPIO.input(GPIO_ECHO) == 1:
            self.last_update = time.time()

        # time difference between start and arrival
        time_elapsed = self.last_update - start_time
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (time_elapsed * 34300) / 2

        return distance


if __name__ == '__main__':
    try:
        sensor = DistanceSensor(GPIO_TRIGGER, GPIO_ECHO)
        while True:
            dist = sensor.get_distance()
            print("Measured Distance = %.1f cm" % dist)
            time.sleep(1)

        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()
