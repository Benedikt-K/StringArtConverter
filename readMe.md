# StringArtConverter

Turn any image into **string art**: load a photo, preprocess (grayscale, edges, tone), and generate a sequence of nails-to-nails lines. Includes live preview, export, and a responsive UI (Qt/PySide6).

---

## ✨ Features

- **Clean desktop UI** (PySide6)  
- **Preprocessing pipeline**  
- **Greedy solver**  
- **Renderer**  
- **Export**  

---

## 🧱 Project Structure

StringArtConverter/
├─ main.py # app entrypoint
├─ StringArtConverter/
│ ├─ main_ui.py # Qt window, buttons, worker, image preview
│ ├─ pin_solver.py # solver + masks + scoring
│ ├─ simulate.py # renderer for existing path
│ ├─ preprocessing.py # grayscale/edges/target/resize/remove bg
│ ├─ worker.py # QThread wrapper (progress, result)
│ └─ init.py
├─ requirements.txt
└─ README.md

## 🚀 Quick Start

### 1) Create & activate a Conda env 

```bash
conda create -n StringArtConverter python=3.10 -y
conda activate StringArtConverter
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
pip install rembg onnxruntime==1.18.0  # or the latest stable CPU-only
```

If you see a onnxruntime_providers_cuda.dll / cublasLt64_12.dll error, uninstall GPU ORT and install CPU:

```bash
pip uninstall onnxruntime-gpu -y
pip install onnxruntime
```

### 3) Run
```bash
python main.py
```