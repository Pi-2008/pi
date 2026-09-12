import serial
import pyautogui

arduino = serial.Serial("COM11", 9600, timeout=1)

while True:
    command = arduino.readline().decode(errors="ignore").strip()

    if command == "PRESS":
        pyautogui.press("space")
        print("BUTTON PRESSED")        