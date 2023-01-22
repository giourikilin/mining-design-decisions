import kivy.app
import kivy.properties
import kivy.uix.boxlayout

from . import content_window
from . import benchmark_window
from . import single_file_analysis_window
from . import multi_file_analysis_window


class ManagerWindow(kivy.uix.boxlayout.BoxLayout):

    content_frame = kivy.properties.ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__current_window = None

    def goto_benchmark(self):
        self.__switch_content(
            content_window.NullWindow('Benchmark')
        )

    def goto_single_file_analysis(self):
        self.__switch_content(
            content_window.NullWindow('Single File Analysis')
        )

    def goto_multi_file_analysis(self):
        self.__switch_content(
            multi_file_analysis_window.MultiFileAnalysisWindow()
        )

    def __switch_content(self, window: content_window.ContentWindow):
        if self.__current_window is not None and window.locked:
            return
        if self.__current_window is not None:
            self.content_frame.remove_widget(self.__current_window.get_window())
        self.content_frame.add_widget(window.get_window())
        self.__current_window = window


class DLManagerApp(kivy.app.App):

    def build(self):
        return ManagerWindow()
