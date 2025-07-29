import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logging.getLogger("src").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

import sys
from PyQt5 import QtWidgets
from src.frasta_gui import MainWindow

def run():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

run()
