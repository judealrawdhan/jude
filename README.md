import RPi.GPIO as GPIO
import time

# تعريف أرقام الـ GPIO لكل لون في الإشارة
RED_LIGHT = 17
YELLOW_LIGHT = 27
GREEN_LIGHT = 22

# إعداد الـ GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LIGHT, GPIO.OUT)
GPIO.setup(YELLOW_LIGHT, GPIO.OUT)
GPIO.setup(GREEN_LIGHT, GPIO.OUT)

# تشغيل الضوء الأحمر افتراضيًا عند بدء التشغيل
GPIO.output(RED_LIGHT, GPIO.HIGH)
GPIO.output(YELLOW_LIGHT, GPIO.LOW)
GPIO.output(GREEN_LIGHT, GPIO.LOW)

def set_traffic_light(color):
    GPIO.output(RED_LIGHT, GPIO.LOW)
    GPIO.output(YELLOW_LIGHT, GPIO.LOW)
    GPIO.output(GREEN_LIGHT, GPIO.LOW)

    if color == "red":
        GPIO.output(RED_LIGHT, GPIO.HIGH)
    elif color == "yellow":
        GPIO.output(YELLOW_LIGHT, GPIO.HIGH)
    elif color == "green":
        GPIO.output(GREEN_LIGHT, GPIO.HIGH)

def turn_green_light():
    print("🚦 تشغيل الإشارة الخضراء!")
    set_traffic_light("green")
    time.sleep(5)  # إبقاء الضوء الأخضر لمدة 5 ثوانٍ

    print("🟡 تحويل الإشارة إلى الأصفر!")
    set_traffic_light("yellow")
    time.sleep(2)  # بقاء اللون الأصفر لفترة وجيزة

    print("🔴 إعادة الإشارة إلى الأحمر!")
    set_traffic_light("red")
