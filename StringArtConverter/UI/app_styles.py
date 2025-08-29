APP_STYLES = """
/* ----------------- Global base ------------------ */
QMainWindow {
    background: #0b0c10;
    color: #e6edf3;
}
QWidget {
    background: #0b0c10;
    color: #e6edf3;
}

/* ------------------- Scroll area ----------------------- */
QScrollArea {
    background: #0b0c10;
    border: 1px solid #2b2f36;
    border-radius: 10px;
}
QScrollArea > QWidget {
    background: transparent;
}
QScrollArea > QWidget > QWidget {
    background: #0b0c10;
}

/* scrollbars */
QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 0px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #2b2f36;
    border-radius: 6px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #3b82f6;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
    border: none;
}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
}

QScrollBar:horizontal {
    background: transparent;
    height: 12px;
    margin: 0px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #2b2f36;
    border-radius: 6px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #3b82f6;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    border: none;
}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
}

/* ---------------- Group boxes --------------- */
QGroupBox {
    margin-top: 14px;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 12px;
    background-color: #0b0c10;
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

/* --------------- Labels/inputs ------------------ */
QLabel, QCheckBox, QSpinBox, QDoubleSpinBox {
    color: #e6edf3;
    font-size: 13px;
}

/* ------------------------- Sliders ------------------------------- */
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

/* ------------------------ Buttons -------------------------- */
QPushButton {
    background: #1f6feb;
    color: #ffffff;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 6px 10px;
    min-height: 18px;
}
QPushButton:hover {
    background: #2a7ef0;
}
QPushButton:pressed {
    background: #1a63d8;
}
QPushButton:checked{
    background: #215fd1;
    border-color: #93c5fd;
}

QPushButton:disabled {
    background: #1a2230;
    color: #9aa4ad;
    border: 1px solid #2b2f36;
    border-radius: 10px;
}

/* ----------------- Convert button ------------------ */
QPushButton#btn_convert {
    background-color: #1f6feb;
    color: #ffffff;
    border: 1px solid #2b2f36;
    border-radius: 18px;
    padding: 10px 16px;
    font-weight: 600;
}
QPushButton#btn_convert:hover  { background: #2a7ef0; }
QPushButton#btn_convert:pressed{ background: #1a63d8; }
QPushButton#btn_convert:disabled{
    background-color: #334155; color: #9aa4ad;
}

/* ----------------- Progress bar ------------------------------- */
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

/* ---------------- checkboxes ------------------------ */
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

/* ------------------------------------ Cards --------------------------- */
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

/* ----------------- Tooltip --------------- */
QToolTip {
    background-color: #111318;
    color: #e6edf3;
    border: 1px solid #3b82f6;
    padding: 8px 10px;
    border-radius: 8px;
    font-size: 16px;
}

/* ---------------------- Combobox ----------------------- */
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

/* ------------------------ SpinBoxes ----------------------- */
QLineEdit,
QSpinBox,
QDoubleSpinBox {
    background: #0f1117;
    color: #e6edf3;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 8px 10px;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
}

/* Hover + focus */
QLineEdit:hover,
QSpinBox:hover,
QDoubleSpinBox:hover {
    border-color: #3a3f47;
    background: #12151d;
}

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 2px solid #1f6feb;
    background: #12151d;
}

/* Disabled */
QLineEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled {
    background: #121318;
    color: #9aa4ad;
    border-color: #2b2f36;
}

/* buttons */
QSpinBox::up-button,
QSpinBox::down-button,
QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button {
    background: transparent;
    border: none;
    width: 16px;
    margin: 2px 4px;
    border-radius: 6px;
}

QSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover {
    background: #1c2330;
}

QSpinBox::up-button:pressed,
QSpinBox::down-button:pressed,
QDoubleSpinBox::up-button:pressed,
QDoubleSpinBox::down-button:pressed {
    background: #202836;
}

/* arrows */
QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {
    width: 10px; height: 10px;
}
QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {
    width: 10px; height: 10px;
}

/* -------- Menu bar  -------- */
QMenuBar {
    background: #0b0c10;
    color: #e6edf3;
    border: none;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 10px;
    margin: 0 2px;
    border-radius: 8px;
}
QMenuBar::item:selected,
QMenuBar::item:pressed {
    background: #111827;
    color: #ffffff;
}

/* -------- Menu ( dropdown) -------- */
QMenu {
    background: #0b0c10;
    color: #e6edf3;
    border: 1px solid #2b2f36;
    padding: 6px 0;
    border-radius: 10px;
}
QMenu::item {
    padding: 6px 14px;
    border-radius: 6px;
}
QMenu::item:selected {
    background: #1f6feb;
    color: #ffffff;
}
QMenu::item:disabled {
    color: #9aa4ad;
    background: transparent;
}

/* separators */
QMenu::separator {
    height: 1px;
    margin: 6px 8px;
    background: #2b2f36;
}

/* submenu arrow */
QMenu::right-arrow {
    width: 10px; height: 10px;
}
"""