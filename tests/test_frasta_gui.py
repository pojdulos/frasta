import numpy as np
import pytest
from PyQt5 import QtWidgets
from src.frasta_gui import MainWindow


@pytest.fixture
def mainwindow(qapp):
    # qapp fixture z conftest.py
    win = MainWindow()
    yield win
    win.close()
    QtWidgets.QApplication.processEvents()

def test_create_circle_mask(mainwindow):
    mask = mainwindow.create_circle_mask((10, 10), (5, 5), 3)
    assert mask.shape == (10, 10)
    assert mask[5, 5]  # środek powinien być True

def test_create_rectangle_mask(mainwindow):
    mask = mainwindow.create_rectangle_mask((10, 10), (5, 5), 4, 4)
    assert mask.shape == (10, 10)
    assert mask[5, 5]  # środek powinien być True

def test_add_to_recent_files(mainwindow, tmp_path):
    test_file = str(tmp_path / "test.csv")
    with open(test_file, "w") as f:
        f.write("x;y;z\n1;2;3\n")
    mainwindow.add_to_recent_files(test_file)
    assert mainwindow.recent_files[0] == test_file

def test_update_recent_files_menu(mainwindow, tmp_path):
    test_file = str(tmp_path / "test.csv")
    mainwindow.recent_files = [test_file]
    mainwindow.update_recent_files_menu()
    assert mainwindow.recent_menu.actions()[0].text() == test_file

def test_save_tabs_no_tabs(mainwindow):
    # Sprawdza, czy funkcja nie rzuca wyjątku, gdy nie ma danych do zapisania
    mainwindow.save_tabs(tabs=None)

def test_load_recent_files(mainwindow):
    mainwindow.recent_files = []
    mainwindow.settings.setValue("recentFiles", ["a", "b"])
    mainwindow.load_recent_files()
    assert mainwindow.recent_files == ["a", "b"]

def test_create_tab_and_load_unknown_format(mainwindow, tmp_path):
    test_file = str(tmp_path / "test.unknown")
    with open(test_file, "w") as f:
        f.write("dummy")
    # Funkcja powinna po prostu zakończyć się bez wyjątku
    mainwindow.create_tab_and_load(test_file)
