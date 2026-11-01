# Webcam Motion Mapping

> **Final Course Output (FCO)** — Real-time webcam-based third-person to first-person motion mapping using **MediaPipe**, **OpenCV**, and mathematical coordinate transformations.

---

## Overview

This project uses a webcam to track **hand and face movements in real time** and map the detected motion into a first-person perspective.

The system uses **MediaPipe** for landmark detection, **OpenCV** for webcam processing, and **NumPy** for mathematical and coordinate transformations.

---

## Requirements

Before running the project, make sure you have:

- **Python 3.x**
- **Git**
- A working **webcam**
- An **internet connection** for the first run if the model files need to be downloaded

### Python Packages

The project automatically installs the required packages through `requirements.txt`:

- MediaPipe
- OpenCV
- NumPy

---

# Quick Start

The easiest way to get the project running is to **clone the repository and run the setup commands**.

### 1. Clone the Repository

Open **Command Prompt** or **PowerShell** and run:

```bash
git clone https://github.com/MitoDoCode/webcam-motion-mapping-fco.git
```

Then enter the project folder:

```bash
cd webcam-motion-mapping-fco
```

---

### 2. Install the Requirements

Run:

```bash
pip install -r requirements.txt
```

This installs:

- `mediapipe`
- `opencv-python`
- `numpy`

---

### 3. Run the Program

Make sure your webcam is connected, then run:

```text
run.bat
```

You can also double-click `run.bat` from File Explorer.

---

##  Running on Arch Linux

If you are using **Arch Linux**, follow these instructions instead of the Windows setup above.

### 1. Install the Required Packages

Open your terminal and run:

```bash
sudo pacman -S python python-pip python-opencv git
```

### 2. Clone the Repository

```bash
git clone https://github.com/MitoDoCode/webcam-motion-mapping-fco.git
cd webcam-motion-mapping-fco
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install the Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Program

Make sure your webcam is connected, then run:

```bash
python fpv_pipeline_faceLandmarker.py
```

---

###  First Run

On the first run, the program may automatically download the required `.task` model files if they are not already present.

The required models include:

- Full Model
- Hand Landmarker
- Face Landmarker

> **Note:** An internet connection is required during the first run if the model files are missing.

---

# 🎮 Controls

| Key | Action |
|:---:|---|
| `Q` | Quit the program |
| `H` | Toggle finger/hand tracking |
| `F` | Toggle Face Landmarker |

### Quick Reference

```text
Q → Quit
H → Toggle Finger Tracking
F → Toggle Face Landmarker
```

---

# Project Structure

After cloning the repository, the project will look similar to:

```text
webcam-motion-mapping-fco/
│
├── fpv_pipeline_faceLandmarker.py
├── requirements.txt
├── run.bat
├── README.md
│
├── *.task
└── ...
```

The `.task` model files should be located in the same directory as:

```text
fpv_pipeline_faceLandmarker.py
```

---

# Troubleshooting

## Git is not recognized

If you get:

```text
'git' is not recognized as an internal or external command
```

make sure Git is installed and available in your system PATH.

You can check your installation with:

```bash
git --version
```

---

## Python is not recognized

Check whether Python is installed:

```bash
python --version
```

If that does not work, try:

```bash
py --version
```

---

## Required packages are missing

Run:

```bash
pip install -r requirements.txt
```

again from inside the project directory.

---

## Model files are missing

Make sure the `.task` files are in the same directory as:

```text
fpv_pipeline_faceLandmarker.py
```

If they are missing, connect to the internet and run the program again so the required models can be downloaded.

---

## Webcam is not detected

Make sure:

- The webcam is connected.
- No other application is currently using the webcam.
- The camera is accessible by Python/OpenCV.
- Windows has permission to access the camera.

---

# Tools & Framework Used

| Tools & Framework | Purpose |
|---|---|
| **Python** | Main programming language |
| **MediaPipe** | Hand and face landmark detection |
| **OpenCV** | Webcam and image processing |
| **NumPy** | Mathematical and coordinate transformations |
| **Git** | Project version control |
| **Batch Script** | Simplified program launching |

---

# Contributors

- [**Nouvel03**](https://github.com/Nouvel03)
- [**MitoDoCode**](https://github.com/MitoDoCode)

---

