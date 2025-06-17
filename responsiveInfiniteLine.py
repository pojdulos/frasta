from PyQt5 import QtCore
import pyqtgraph as pg

class ResponsiveInfiniteLine(pg.InfiniteLine):
    """
    InfiniteLine z płynnym przesuwaniem i throttlingiem przeliczania.
    """
    def __init__(self, update_callback=None, delay_ms=200, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setCursor(QtCore.Qt.SizeHorCursor if self.angle == 90 else QtCore.Qt.SizeVerCursor)

        # Timer do throttlowania update'ów
        self._update_callback = update_callback
        self._update_timer = QtCore.QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._handle_throttled_update)

        self._last_value = self.value()
        self._throttle_delay = delay_ms

        self.sigPositionChanged.connect(self._on_position_changed)

        # Dodatkowy sygnał po zakończeniu przeciągania (jeśli dostępny)
        if hasattr(self, "sigPositionChangeFinished"):
            self.sigPositionChangeFinished.connect(self._on_change_finished)

    def _on_position_changed(self):
        if self._update_callback:
            # Startuj opóźniony update po przesunięciu
            self._update_timer.start(self._throttle_delay)

    def _handle_throttled_update(self):
        value = self.value()
        if value != self._last_value:
            self._last_value = value
            self._update_callback(value)

    def _on_change_finished(self):
        """Natychmiastowe przeliczenie po puszczeniu myszki."""
        self._update_timer.stop()
        value = self.value()
        self._last_value = value
        if self._update_callback:
            self._update_callback(value)
