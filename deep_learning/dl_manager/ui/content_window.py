import abc

import kivy.uix.widget
import kivy.uix.label


class ContentWindow(abc.ABC):

    def __init__(self):
        self.__locked = False

    @property
    def locked(self) -> bool:
        return self.__locked

    def lock(self):
        self.__locked = True

    def unlock(self):
        self.__locked = False

    @abc.abstractmethod
    def get_window(self) -> kivy.uix.widget.Widget:
        pass


class NullWindow(ContentWindow):

    def __init__(self, name):
        super().__init__()
        self.__window = kivy.uix.label.Label(text=name)

    def get_window(self) -> kivy.uix.widget.Widget:
        return self.__window
