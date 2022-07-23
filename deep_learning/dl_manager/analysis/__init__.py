import seaborn
seaborn.set_theme()

from .compare import run_comparison_command
from .plots import run_bar_plot_command, run_plot_attributes_command
from .summarize import run_summarize_command
from .confusion import run_confusion_matrix_command
from .stat_matrix import run_stat_command
