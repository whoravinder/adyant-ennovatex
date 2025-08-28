# UAV Parameter Estimation using ISAC + Generative AI  

This repository contains three components:  
1. **Software Pipeline** → 3GPP-compliant dataset generation, AI model training & evaluation.  
2. **Hardware Pipeline** → ESP32 + HC-SR04 ultrasonic sensor streaming real range data.  
3. **Integrated Pipeline** → Combines hardware + AI estimator + Generative AI predictor with live dashboard.  

---

## 1️⃣ Software Pipeline

### Installation
```bash
git clone https://github.com/your-repo/uav-isac.git
cd uav-isac/software_pipeline
python -m venv uav_env
source ../uav_env/bin/activate      # Linux/Mac
..\uav_env\Scripts\activate         # Windows
pip install -r ../requirements.txt
```

### Usage
1. **Generate Datasets**
```bash
python generate_dataset.py
```
Creates `datasets/adyant_isac_UMi.h5`, `adyant_isac_UMa.h5`, etc.

2. **Train Model**
```bash
python train.py
```
Trains UAVEstimator and saves as `adyant_uav_estimator.pth`.

3. **Evaluate Model**
```bash
python dashboard.py
```
Outputs metrics + plots:  
- **Trajectory (True vs Predicted)**  
- **Time-Series (Range, Velocity, DOA)**  
- **Error Plots (Raw − AI)**  

---

## 2️⃣ Hardware Pipeline

### Hardware Requirements
- ESP32-WROOM Dev board  
- HC-SR04 Ultrasonic Sensor  
- Jumper wires + Breadboard  
- Micro-USB cable  

### Wiring Guide
| HC-SR04 Pin | ESP32 Pin   |
|-------------|-------------|
| VCC (5V)    | 5V          |
| GND         | GND         |
| TRIG        | GPIO5       |
| ECHO        | GPIO18      |

⚠️ Note: HC-SR04 ECHO outputs 5V. Use a **voltage divider (1kΩ + 2kΩ)** to step down to safe 3.3V for ESP32 GPIO.

---

### Installation
1. Install **Arduino IDE** → [Download](https://www.arduino.cc/en/software)  
2. Install **ESP32 board package** in Arduino IDE.  
3. Open and upload:
```bash
hardware_pipeline/esp32_ultrasonic.ino
```
Select `ESP32 Dev Module` + correct COM port → Upload.  

---

### Usage
1. Connect ESP32 via USB.  
2. Open Arduino serial monitor @115200 baud → verify distance values.  
3. Close Arduino IDE serial monitor.  
4. Run Python receiver:
```bash
python data_receiver.py
```
Outputs live **Range (cm)**, **Velocity (cm/s)**, **Synthetic DOA (°)**.  

---

## 3️⃣ Integrated Pipeline

This combines **hardware stream + AI Estimator + Generative Predictor** into one live demo.

---

### Installation
```bash
git clone https://github.com/your-repo/uav-isac.git
cd uav-isac/integrated_pipeline
python -m venv uav_env
source ../uav_env/bin/activate      # Linux/Mac
..\uav_env\Scripts\activate         # Windows
pip install -r ../requirements.txt
```

---

### Usage
1. Flash ESP32 with firmware (`esp32_ultrasonic.ino`).  
2. Connect ESP32 to USB.  
3. Run integrated demo:
```bash
python integrated_hardware_ai.py
```

---

### Outputs
The dashboard shows 3 stacked plots in real time:
1. **Trajectory Plot**  
   - Blue = Raw UAV Path  
   - Red dashed = AI Estimator Path  
   - Green dashed = Predicted Future Path (with ghost trails)  

2. **Time-Series Plot (last 30s)**  
   - AI-estimated Range, Velocity, DOA  

3. **Error Plot (last 30s)**  
   - Difference (Raw − AI) for Range, Velocity, DOA  

---

