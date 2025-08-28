### Our Approach  
1. **End-to-End UAV Tracking Pipeline**  
   - Built a **3GPP TR 38.901 (Sec. 7.9)** compliant simulator → generates realistic UAV signals under **urban, suburban, and rural propagation (UMi,UMa,SMa,RMa)**.
   - Extracted **delay (range), Doppler (velocity), and angular (DOA)** features using **FFT/STFT**.  
   - Designed a **next-gen AI model** (Residual CNN + Attention Fusion) that learns to map noisy ISAC features to accurate UAV parameters.  

2. **Generative AI for Predictive Awareness**  
   - Added a **sequence predictor (RNN)** that doesn’t just estimate — it **forecasts UAV trajectory**.  
   - Provides **proactive awareness**, predicting where the UAV will be in the next few seconds.  

3. **Low-Cost Hardware Prototype**  
   - Deployed on **ESP32 + Ultrasonic sensor** (< ₹400 / $5).  
   - Demonstrates how the **same AI model** can fuse with real sensor data.  

4. **Integrated Dashboard with Plots**  
   - Unified visualization includes:  
     - **Trajectory Plot**: Raw path (blue), AI path (red), Predicted future path (green).  
     - **Time-Series Plots**: Range, Velocity, DOA over time → show how AI smooths noisy data.  
     - **Error Plots**: Difference between raw and AI estimates → prove accuracy.  
    

---

### Why It’s Unique  
**3GPP Compliant + Real Hardware**: Most projects either simulate OR prototype. We bridged both — standards-based simulation + physical sensor demo.  

**Hybrid AI + Generative AI**: Instead of just estimating, our model **anticipates** UAV behavior with Generative AI forecasting.  

**Affordable + Scalable**: Achieves ISAC-style UAV sensing with an **ESP32 + ultrasonic sensor** that anyone can replicate.  

**Explainable Visualization**: Multiple plots (trajectory, time-series, error) make results transparent and verifiable.  

**Social & Security Impact**:
Our solution tries to solve real world problems and build something which is economically and deploys huge impact, our solution is expected to be scalable enough to be deployed on following sectors
- **Defense & Airspace Security** → early UAV detection & prediction.  
- **Disaster Response** → UAVs for rescue, AI predicts paths for coordination.  
- **Smart Cities** → low-cost ISAC sensors monitoring aerial drones.  
