**Problem Statement to the Solution**
# Estimation of UAV Parameters Using Monostatic Sensing in ISAC Scenario 

| *Problem Statement Requirement* | *Our Proposed Solution* |
|-----------------------------------|----------------------------|
| *Estimation of UAV parameters using monostatic sensing in ISAC scenario* | Built an *end-to-end AI pipeline* to estimate UAV *Range, Velocity, DOA* using monostatic ISAC signals. Combines signal processing + deep learning instead of rule-based estimation. |
| *AI-based solution leveraging advanced signal processing and machine learning* | Extracted ISAC features using *FFT (range profile), **STFT (Doppler spectrum), **DOA one-hot encoding. Designed **UAVEstimator (Residual CNNs + Attention Fusion)* to directly predict {Range, Velocity, DOA}. |
| *Utilize 3GPP TR 38.901-j00 (Rel-19) Section 7.9 channel model* | Implemented a *dataset generator* following *3GPP TR 38.901 Sec. 7.9. Simulated **UMi, UMa, RMa, SMa* scenarios with LOS/NLOS, Doppler, multipath fading. Stored datasets in .h5 format (adyant_isac_*.h5). |
| *ISAC Scenario* | Since ISAC is unified for sensing and communication, we have used a near cheap alternative to it which is ultrasoic sensor. Ultrasonic sensor is also unified system with Single sensor for sending and recieving the signals|
| *Design models that extract UAV parameters under channel conditions* | Trained UAVEstimator on generated datasets. AI model extracts *Range, Velocity, DOA* under realistic channel conditions. Added *Generative AI PredictorRNN* for future UAV path forecasting. |
| *Practical demonstration & interpretability* | Integrated *ESP32 + Ultrasonic sensor* for hardware validation. Built a *live dashboard* showing: <br> - Trajectory plots (Raw vs AI vs Predicted) <br> - Time-series plots (Range, Velocity, DOA) <br> - Error plots (Raw − AI). |
