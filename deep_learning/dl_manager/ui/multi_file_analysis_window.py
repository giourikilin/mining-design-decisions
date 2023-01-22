import glob
import os.path
import time
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import tkinter.simpledialog

import kivy.clock
import kivy.lang
import kivy.properties
import kivy.uix.boxlayout
import kivy.uix.widget

import matplotlib.pyplot as pyplot
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

from . import content_window
from . import run_analysis


directory = os.path.split(__file__)[0]
kv_file = os.path.join(
    directory, 'multi_file_analysis_frame.kv'
)
kivy.lang.Builder.load_file(kv_file)


class MultiFileAnalysisFrame(kivy.uix.boxlayout.BoxLayout):

    plotting_window = kivy.properties.ObjectProperty(None)
    files_list = kivy.properties.StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__files = []
        self.__attributes = []
        self.__perform_trimming = False
        self.__min_index = None
        self.__patience = None
        self.__last_update = -1
        self.__current_plot = None

    def select_files(self):
        root = tkinter.Tk()
        root.withdraw()
        files = tkinter.filedialog.askopenfilenames(title='Choose Files to Analyze')
        self.__files = sorted(files)
        self.files_list = '\n'.join(self.__files)

    def glob_select(self):
        root = tkinter.Tk()
        root.withdraw()
        source_directory = tkinter.filedialog.askdirectory(title='Choose Source Directory')
        pattern = tkinter.simpledialog.askstring(title='Enter Pattern',
                                                 prompt='Specify The Globbing Pattern')
        files = glob.glob(pattern, root_dir=source_directory)
        self.__files = sorted(os.path.join(source_directory, file) for file in files)
        self.files_list = '\n'.join(self.__files)

    def clear_files(self):
        self.__files = []
        self.files_list = ''

    def enter_patience(self, patience_as_string):
        if not patience_as_string:
            self.__patience = None
        else:
            self.__patience = int(patience_as_string)

    def enter_min_index(self, index_as_string):
        if not index_as_string:
            self.__min_index = None
        else:
            self.__min_index = int(index_as_string)

    def enter_attributes(self, attribute_input):
        if not attribute_input:
            self.__attributes = []
        else:
            self.__attributes = [
                attribute.strip() for attribute in attribute_input.splitlines()
            ]

    def set_trimming(self, state):
        self.__perform_trimming = state == 'down'

    def create_plot(self):
        self.__update_plot()

    def __update_plot(self):
        self.__last_update = time.time()
        if not self.__files:
            return
        if not self.__attributes:
            return
        if self.__perform_trimming:
            if self.__min_index is None or self.__patience is None:
                return
        self.__make_plot()

    def __make_plot(self):
        fig, axes = run_analysis.run_bar_plot_command(
            files=self.__files,
            attributes=self.__attributes,
            trim=self.__perform_trimming,
            patience=self.__patience,
            min_index=self.__min_index,
            include_maxima=False,
            tolerance=0.0
        )
        if self.__current_plot is not None:
            self.plotting_window.remove_widget(self.__current_plot)
        self.__current_plot = FigureCanvasKivyAgg(fig)
        self.plotting_window.add_widget(self.__current_plot)


class MultiFileAnalysisWindow(content_window.ContentWindow):

    def __init__(self):
        super().__init__()
        self.__widget = MultiFileAnalysisFrame()

    def get_window(self) -> kivy.uix.widget.Widget:
        return self.__widget
