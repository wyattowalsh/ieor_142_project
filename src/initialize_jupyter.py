import multiprocessing
import warnings
from IPython.display import display, HTML

# Configure notebook display and warnings
display(HTML("<style>.container { width:90% }</style>"))
display(HTML("<style>div.output_scroll { height: 48em; }</style>"))
display(HTML("""<style> .output_png {display: table-cell;
        text-align: center;
        vertical-align: middle;
    }</style>"""))
CSS = """
div.cell:nth-child(3) .output {
    flex-direction: row;
}
"""
HTML('<style>{}</style'.format(CSS))

warnings.filterwarnings('ignore')


