
from pathlib import Path

from IPython.display import display, Markdown


version_info = (0, 0, 1)
__version__ = '.'.join(map(str, version_info))

github_url = "https://github.com/ContextLab/sherlock-topic-model-paper/tree/master/code/sherlock_helpers"
pkg_dir = Path(__file__).resolve().parent
message = Markdown(
    "Helper functions and variables used across multiple notebooks can be "
    f"found in `{pkg_dir}`, or on GitHub, [here]({github_url}).<br />You can "
    "also view source code directly from the notebook with:<br /><pre>    "
    "from sherlock_helpers.functions import show_source<br />    show_source(foo)"
    "</pre>"
)

# Modified: removed code to check for use in ipython notebook

