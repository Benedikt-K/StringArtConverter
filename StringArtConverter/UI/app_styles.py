APP_STYLES = """
/* Global base */
QMainWindow {
    background: #0b0c10;
    color: #e6edf3;
}
QWidget {
    background: #0b0c10;         /* <- default to dark everywhere */
    color: #e6edf3;
}

/* Scroll area needs both the abstract and viewport styled */
QAbstractScrollArea,
QAbstractScrollArea::viewport,
QScrollArea,
QScrollArea QWidget {
    background: #0b0c10;
}

/* Right panel explicit (in case you name the container) */
#RightPanel {
    background: #0b0c10;
}

/* Card-like groups */
QGroupBox {
    margin-top: 14px;
    border: 1px solid #2b2f36;
    border-radius: 10px;
    padding: 12px;
    background-color: #1c1f26;   /* lighter card on dark surface */
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

/* primary CTA button */
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

/* Modern circular checkboxes with strong contrast */
QCheckBox {
    spacing: 10px;
    color: #e6edf3;
    font-size: 13px;
}

/* base (unchecked) */
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #2b2f36;      /* visible ring */
    background: #0f1116;            /* inner fill */
    margin-top: 2px;
}

/* hover while unchecked */
QCheckBox::indicator:hover {
    border-color: #5a6472;          /* a bit brighter ring on hover */
    background: #11151c;
}

/* CHECKED: high-contrast border + solid fill */
QCheckBox::indicator:checked {
    background: #1f6feb;            /* brand fill */
    border: 2px solid #0b2e7a;      /* darker ring for higher contrast */
}

/* hover while CHECKED: brighter fill + lighter border */
QCheckBox::indicator:checked:hover {
    background: #2a7ef0;            /* lift the fill on hover */
    border-color: #1348b5;          /* lighten the ring a bit */
}

/* pressed feedback (optional) */
QCheckBox::indicator:checked:pressed {
    background: #1a63d8;
    border-color: #0a2770;
}
QCheckBox::indicator:pressed {
    background: #0e131b;
    border-color: #4b5563;
}

/* indeterminate (optional) */
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
"""