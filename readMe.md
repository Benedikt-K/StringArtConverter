# StringArtConverter

**-- CURRENTLY STILL IN DEVELOPMENT --**

Turn any image into **string art**:
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

- **Desktop UI**
- **Preprocessing pipeline**
- **Greedy Solver**
- **Renderer**
- **Export**

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
pip install rembg onnxruntime
```

## Option A (recommended):

start with:
```
python main.py
```

then use User Interface for parameter selection.

## Option B:

use Command Line Interface with parameters:


--specify later--

# Structure

```
StringArtConverter/
├─ main.py
├─ StringArtConverter/
│  ├─ UI/
│  │  ├─ app_styles.py
│  │  ├─ main_window.py
│  │  ├─ sliders.py
│  │  ├─ ui_utils.py
│  │  └─ settings.json
│  ├─ preprocessing.py
│  ├─ previewer.py
│  ├─ solver.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```
# License

Distributed under the MIT License.

# Topics

`python`, `pyside6`, `image-processing`, `string-art`, `computer-vision`, `onnx`