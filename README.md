# Remote_Heart_Rate_Using_Deep_Learning

The main objective of this project is to compare four most recent deep learning-based Image Processing Algorithms for Remote Heart Rate. The method used in all the all the four models is remote PPG. 

Physiological measurements are widely used to determine a person’s health condition. Photoplethysmography (PPG) is a physiological measurement method that is
used to detect volumetric changes in blood in vessels beneath the skin. Medical devices
based on PPG have been introduced to measure different physiological measurements
including heart rate (HR), respiratory rate, heart rate variability (HRV), oxyhemoglobin
saturation, and blood pressure . Due to its low cost and non-invasive nature, PPG
is utilized in many devices such as finger pulse oximeters, sports bands, and wearable
sensors.

The public domain UBFC dataset is used to compare the performance of these deep learning methods for heart rate measurement.  UBFC dataset comprised of 42 videos of 42 individuals corresponding to heart rate label. Each image frame is of size 640×480 and is captured in RGB channels. For training all the model, I used 80% of the extracted input image frames from UBFC dataset videos and 20% for testing all the models. The Models are compared based on accuracy and loss.
