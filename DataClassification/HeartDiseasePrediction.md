## Dataset
* [UCI Heart disease] (https://archive.ics.uci.edu/ml/datasets/heart+Disease) This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.

* https://github.com/mkader/ML.NET/blob/main/Dataset/HeartTraining.csv
## Problem 
* Predicting the presence of heart disease based on 14 attributes. Take all 14 columns, 13 are feature columns, the num (Label) column to predict
* Attribute Information:
  1. (age) - Age
  1. (sex) - (1 = male; 0 = female)
  1. (cp) chest pain type -- Value 1: typical angina -- Value 2: atypical angina -- Value 3: non-anginal pain -- Value 4: asymptomatic
  1. (trestbps) - resting blood pressure (in mm Hg on admission to the hospital)
  1. (chol) - serum cholestoral in mg/dl
  1. (fbs) - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
  1. (restecg) - esting electrocardiographic results
      1. Value 0: normal
      2. Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
      3. Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
  1. (thalach) - maximum heart rate achieved
  1. (exang) - exercise induced angina (1 = yes; 0 = no)
  1. (oldpeak) - ST depression induced by exercise relative to rest
  1. (slope) - the slope of the peak exercise ST segment -- Value 1: upsloping -- Value 2: flat -- Value 3: downsloping
  1. (ca) - number of major vessels (0-3) colored by flourosopy
  1. (thal) - 3 = normal; 6 = fixed defect; 7 = reversible defect
  1. (num) - (the predicted attribute) diagnosis of heart disease (angiographic disease status)
      1. Value 0: < 50% diameter narrowing
      2. Value 1: > 50% diameter narrowing
      1. Predicts the presence of heart disease in the patient with integer values from 0 to 4
      2. Experiments with the Cleveland database (dataset used for this example) have concentrated on simply attempting to distinguish presence (value 1) from absence (value 0).
  
