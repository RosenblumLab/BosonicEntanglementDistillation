import qutip
from matplotlib import pyplot as plt

from State import State


class PlotUtils:
    @staticmethod
    def plot_wigner_function_comparison(initial_state: State, new_state: State):
        fig, axes = plt.subplots(1, 2)
        fig.set_figwidth(10)
        fig.set_figheight(4)

        qutip.plot_wigner(initial_state, fig, axes[0], colorbar=True)
        PlotUtils.edit_graph(axes[0], "<Q>", "<P>", "Before")

        qutip.plot_wigner(new_state, fig, axes[1], colorbar=True)
        PlotUtils.edit_graph(axes[1], "<Q>", "<P>", "After")

        plt.show()

    @staticmethod
    def edit_graph(axes, xlabel, ylabel, title=None, legend=None):
        LABEL_FONT_SIZE = 18
        TICKS_FONT_SIZE = 12
        TITLE_FONT_SIZE = 20
        LEGEND_FONT_SIZE = 10

        axes.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        plt.tight_layout()

        plt.xticks(fontsize=TICKS_FONT_SIZE)
        plt.yticks(fontsize=TICKS_FONT_SIZE)

        if title is not None:
            axes.set_title(title, fontsize=TITLE_FONT_SIZE)

        if legend is not None:
            axes.legend(legend, loc="upper left", fontsize=LEGEND_FONT_SIZE)
