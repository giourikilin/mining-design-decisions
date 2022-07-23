import kivy
kivy.require('2.1.0')

from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')
Config.set('graphics', 'borderless', '0')
Config.write()

from .main_app import DLManagerApp


def start_ui():
    DLManagerApp().run()
