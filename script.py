

import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say("Hello! This is a test.")
engine.runAndWait()
