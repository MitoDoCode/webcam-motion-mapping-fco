# How to Run

Follow the steps below to set up and run the program.

## 1. Download the Repository

Download **all files** from this repository and place them in the **same folder**.

Your folder should contain the main Python script, model files, `requirements.txt`, and `run.bat`.

---

## 2. Install the Requirements

Make sure **Python** is installed on your computer.

### Install the Required Packages

Open a Command Prompt inside the project folder.

**Easy way:**

1. Open the project folder.
2. Click the folder's address bar.
3. Type `cmd`.
4. Press **Enter**.
5. Run:

```bash
pip install -r requirements.txt
```

This will install the required Python packages:

- **MediaPipe**
- **OpenCV**
- **NumPy**

---

## 3. Check the Model Files

Make sure all required `.task` model files are located in the **same folder** as:

```text
fpv_pipeline_faceLandmarker.py
```

The model files are required for the hand and face tracking features.

> **Note:** If the model files are not already present, the program can download the **Full Model**, **Hand**, and **Face Landmarker** model files automatically when you run it for the first time.

---

## 4. Run the Program

Connect a **webcam** to your computer.

Then, run the program by double-clicking:

```text
run.bat
```

The program should start and use your webcam for tracking.

### First Run

During the first run, the program may automatically download the required model files if they are missing.

The models include:

- Full Model
- Hand Landmarker
- Face Landmarker

Make sure you have an **internet connection during the first run** if the required model files are not already in the project folder.

---

## 5. Program Controls

Once the program is running, use the following keyboard controls:

| Key | Function |
|-----|----------|
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

## Troubleshooting

If the program does not start:

1. Make sure **Python is installed**.
2. Run the requirements installation again:

```bash
pip install -r requirements.txt
```

3. Check that the `.task` model files are in the correct folder.
4. Make sure a **webcam is connected** and accessible.
5. If this is the first run, make sure you have an **internet connection** so the required models can be downloaded.
