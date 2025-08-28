# UAV Parameter Estimation using ISAC + Generative AI  

This repository contains three components:  
1. **Software Pipeline** → 3GPP-compliant dataset generation, AI model training & evaluation.  
2. **Hardware Pipeline** → ESP32 + HC-SR04 ultrasonic sensor streaming real range data.  
3. **Integrated Pipeline** → Combines hardware + AI estimator + Generative AI predictor with live dashboard.  
**Note : Hardware and Software components are independent of each other and are combined in Integrated Pipeline**
---

## 1️⃣ Software Pipeline

### Installation
```bash
Cloning the repo: git clone https://github.com/whoravinder/adyant-ennovatex.git
Change your working directory to src folder
Run the following PIP Command:
pip install -r requirements.txt
```
### Software Directory
After cloning, change your working directory to src-> software.  
### Usage
1. **Generate Datasets**
```bash
python generate_dataset.py
```
Creates `datasets/adyant_isac_UMi.h5`, `adyant_isac_UMa.h5`, etc.<br>
Note: No need to run this file as there is already a dataset folder inside software directory, it includes all the datasets used however this command can help regenerate the dataset or if there is no dataset folder then it will generate the datasets. 

2. **Train Model**
```bash
python train.py

you may find the following line
torch.save(model.state_dict(), "adyant_uav_estimator.pth")
Here you can change *adyant_uav_estimator* to your desired name
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

## 2️⃣ Hardware Pipeline [If you have ESP32 WROOM and HC-SR04 Ultrasonic sensor then you may use Hardware else software is enough to demonstrate the solution]

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
hardware/esp32_ultrasonic.ino
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
Outputs live distance from Ultrasonic sensor

```bash
python hardware_demo.py
```
Outputs live **Range (cm)**, **Velocity (cm/s)**, **Synthetic DOA (°)**. 

---

## 3️⃣ Integrated Pipeline

This combines **hardware stream + AI Estimator + Generative Predictor** into one live demo.

---



### Usage
1. Flash ESP32 with firmware (`esp32_ultrasonic.ino`).  
2. Connect ESP32 to USB.  
3. Run integrated demo:
```bash
python integrated.py
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

Note : <br>
Integrated Folder have two important files one is the ai model trained in software pipeline and another is `ai_model.py` file this are required files to run the integrated pipeline

