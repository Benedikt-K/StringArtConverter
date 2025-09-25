# StringArtConverter

Turn any image into **string art**:<br>
Load a photo, preprocess and generate a sequence of nail-to-nail lines.

<table>
  <tr>
    <td><img src="assets/input.jpg" alt="Original" width="260"></td>
    <td align="center"><h1>&rarr;</h1></td>
    <td><img src="assets/output.png" alt="String Art Result" width="260"></td>
  </tr>
  <tr>
    <td align="center"><em>Before</em></td>
    <td></td>
    <td align="center"><em>After</em></td>
  </tr>
</table>


# Features

- **Desktop UI (PySide6)**
- **Preprocessing pipeline (background/face recognition)**
- **Line Generation (greedy solver)**
- **Preview Renderer**
- **Guided pin-by-pin build of the image**

## Pin-by-pin

You can save/load your generated path and from that start a guided session. 
<br><br>
This session shows, which pin connection is the one you need to do next and gives a preview with the current step marked in blue.

<table>
  <tr>
    <td><img src="assets/pin-by-pin.png" alt="screenshot while in pin-by-pin" width="1000"></td>
  </tr>
  <tr>
    <td align="center"><em>UI while in pin-to-pin mode</em></td>
  </tr>
</table>

# Quick start

```
git clone https://github.com/Benedikt-K/StringArtConverter.git
cd StringArtConverter
```

## Conda / pip

```
conda create -n StringArtConverter python=3.10 -y
conda activate StringArtConverter
pip install -r requirements.txt
```

## Start the app with::

```
python main.py
```

Then use User Interface for parameter selection, conversion and all of the rest.

# Structure

```
StringArtConverter/
├─ main.py                  # file to start app from
├─ StringArtConverter/
│  ├─ UI/
│  │  ├─ app_styles.py      # CSS Sytle sheet
│  │  ├─ main_window.py     # UI code for main Window
│  │  ├─ sliders.py         # Custom slider class
│  │  ├─ ui_utils.py
│  │  ├─ workers.py         # Workers so app remains responsive
│  │  └─ settings.json      # Setting presets
│  ├─ preprocessing.py      # Image preprocessing for better results
│  ├─ previewer.py          # Renderer
│  ├─ solver.py             # String-Art-Generating algorithm
│  └─ utils.py
├─ cli.py                   # alternate option to get result via CLI (outdated)
├─ requirements.txt
└─ README.md
```
# License

Distributed under the MIT License.

# Topics

`python`, `pyside6`, `image-processing`, `string-art`, `string-art-generator`, `computer-vision`