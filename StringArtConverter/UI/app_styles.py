APP_STYLES = """
/* Global base */
QMainWindow {
    background: #0b0c10;
    color: #e6edf3;
}
QWidget {
    background: #0b0c10;
    color: #e6edf3;
}

/* Scroll area */
QAbstractScrollArea,
QAbstractScrollArea::viewport,
QScrollArea,
QScrollArea QWidget {
    background: #0b0c10;
}

/* Group boxes */
QGroupBox {
    margin-top: 14px;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 12px;
    background-color: #1c1f26;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    background: transparent;
}

/* Labels/inputs */
QLabel, QCheckBox, QSpinBox, QDoubleSpinBox {
    color: #e6edf3;
    font-size: 13px;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #2b2f36;
    height: 6px;
    border-radius: 3px;
    background: #2b2f36;
}
QSlider::handle:horizontal {
    background: #1f6feb;
    border: none;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #1f6feb;
    border-radius: 3px;
}

/* Buttons */
QPushButton {
    background: #1f6feb;
    color: white;
    border: 0;
    padding: 8px 12px;
    border-radius: 8px;
}
QPushButton:disabled {
    background: #334155;
    color: #9aa4ad;
}

/* Progress bar */
QProgressBar {
    background: #111318;
    border: 1px solid #2b2f36;
    border-radius: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #1f6feb;
    border-radius: 8px;
}

/* Primary convert button */
QPushButton#btn_convert {
    background-color: #1f6feb;
    color: #ffffff;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}
QPushButton#btn_convert:disabled {
    background-color: #334155;
    color: #9aa4ad;
}

/* checkboxes */
QCheckBox {
    spacing: 10px;
    color: #e6edf3;
    font-size: 13px;
}

/* unchecked */
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #2b2f36;
    background: #0f1116;
    margin-top: 2px;
}

/* hover while unchecked */
QCheckBox::indicator:hover {
    border-color: #5a6472;
    background: #11151c;
}

/* checked */
QCheckBox::indicator:checked {
    background: #1f6feb;
    border: 2px solid #0b2e7a;
}

/* hover while checked */
QCheckBox::indicator:checked:hover {
    background: #2a7ef0;
    border-color: #1348b5;
}

/* pressed feedback */
QCheckBox::indicator:checked:pressed {
    background: #1a63d8;
    border-color: #0a2770;
}
QCheckBox::indicator:pressed {
    background: #0e131b;
    border-color: #4b5563;
}

/* indeterminate */
QCheckBox::indicator:indeterminate {
    border: 2px solid #8ab4f8;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #1f6feb, stop:1 #0f1116);
}

/* disabled */
QCheckBox::indicator:disabled {
    border: 2px solid #3a3f46;
    background: #171a20;
}
QCheckBox:disabled {
    color: #9aa4ad;
}

/* Cards */
#CardGroup {
    background: #0b0c10;
    border: 1px solid #2b2f36;
    border-radius: 10px;
}
#CardTitle {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
}
#HelpBadge {
    border: 1px solid #3b82f6;
    border-radius: 11px;
    background: #0b0c10;
    color: #3b82f6;
    font-weight: 700;
    padding: 0;
}
#HelpBadge:hover {
    background: #1f6feb;
    color: #ffffff;
    border-color: #93c5fd;
}
#HelpBadge:pressed {
    background: #194fb6;
}

/* Tooltip */
QToolTip {
    background-color: #111318;
    color: #e6edf3;
    border: 1px solid #3b82f6;
    padding: 8px 10px;
    border-radius: 8px;
    font-size: 16px;
}

/* Combobox */
QComboBox {
    background: #111318;
    color: #e6edf3;
    border: 1px solid #2b2f36;
    border-radius: 8px;
    padding: 6px 34px 6px 10px;
    min-height: 30px;
}
QComboBox:hover { 
    border-color: #3b82f6;
}
QComboBox:disabled {
    background: #0e1014;
    color: #9aa4ad;
    border-color: #1e2128;
}

/* Right "arrow area" */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: 1px solid #2b2f36;
    background: #111318;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}

/* Popup list */
QComboBox QAbstractItemView {
    background: #0b0c10;
    color: #e6edf3;
    border: 1px solid #2b2f36;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    outline: 0;
    padding: 4px;
}
QComboBox QAbstractItemView::item {
    padding: 6px 8px; 
}
QComboBox QAbstractItemView::item:hover {
    background: #111827;
}
"""