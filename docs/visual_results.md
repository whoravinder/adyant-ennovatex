# Software(AI Model) Prototype Screenshots
**1.Predicted Range, Velocity and DOA Plots** <br>
![Software 1](images/soft1.png)<br>
This figure shows predicted values in orange and actual values in blue by AI model, it is very clear through image that model is working well and have good learning on the provided dataset.

**2.Evaluation Measures and True v/s Predicted Table** <br>
![Software 2](images/soft2.png)<br>
This figure shows Range Root Mean Square Error, Velocity Mean Absolute Error and DOA MAE score. It is clear by image and that these scores are very less means model is working well. Another important thing in the figure is that true values and predicted values are very near to each other again proving the model has learned well and have high accuracy.

**3.Predicted v/s Actual Path(UAV Trajectory)** <br>
![Software 3](images/soft3.png)<br>
This figure shows UAV Trajectory where red is of predicted and blue is of original. 

# Hardware Prototype Screenshot
**Hardware recieving values from Ultrasonic Sensor**
![Software 3](images/hard1.png)<br>
Note: Only the Distance is captured from the ultrasonic sensor, velocity is calculated from distance and time and DOA is induced synthetically.

# Integrated Prototype Screenshot
![Software 3](images/inti1.png)<br>
While recieving the real time data from the model here are three plots showing the predicted future path, AI Estimated Parameter, Ai Estimation Error. However due to limited resources and cheap hardware the accuracy of predicted path is slight lower but we tried to demonstrate it perfectly.
