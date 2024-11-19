## Dataset explanation
* The <a href="housing.csv">dataset</a> a slightly modified version of the <a href="https://www.kaggle.com/datasets/vikrishnan/boston-house-prices">Boston House Prices dataset (Original)</a>.
* The Boston House prices dataset aims to predict the median price (in 1972) of a house in one of 506 towns or villages near Boston.
* Each of the 506 data items has 12 predictor variables—11 numeric and 1 Boolean.
* Original dataset - the values are separated by spaces and no headers.
* Edited dataset (Housing.csv) - The spaces replaced by commas and added headers(column names) on the first row.
* Dataset columns
  1. CRIM: Indicates the crime rate by town per capita.
  1. ZN: Reflects the proportion of residential land for lots over 25,000 sq. ft.
  1. INDUS: Specifies the proportion of nonretail business acres per town.
  1. CHAS: Is the Charles River variable (1 if tract bounds river; 0 otherwise).
  1. NOX: Indicates the nitric oxide concentration (parts per ten million).
  1. RM: Indicates the average number of rooms per dwelling.
  1. AGE: Indicates the proportion of owner-occupied units built before 1940.
  1. DIS: Specifies the distances to five employment centers in Boston.
  1. RAD: Indicates the accessibility to highways.
  1. TAX: Indicates the full-value property-tax rate per $10,000.
  1. PTRATIO: Indicates the pupil-teacher ratio by town.
  1. MEDV: Represents the median value of owner-occupied homes in thousands of dollars.
* The goal is to use the Value Prediction scenario using the modified dataset to predict the value of MEDV based on the values represented by the other columns.

## Creating the model
* Create Console application -> select Value prediction scenario ->  input (housing.csv) ->  Select "MEDV" the Column to predict (Label) dropdown ->  Start training -> Training results
* Training data used by this model is FASTFORESTRegression based on decision trees.
* ![image](https://github.com/user-attachments/assets/5c3d0582-ab0b-47f6-878f-f3e6d032362f)
* Next, Evaluate -> The test input values are prefilled with data inferred by Model Builder -> Click Predict (display the prediction results) ->  Consume the model
* ![image](https://github.com/user-attachments/assets/d8c48cf8-84de-49b2-acb3-554e7c9fd314)
* Copy the code snippet and consume it directly from the project
```csharp
//2.Value Prediction
//Load sample data
var sampleData = new ValuePredictionModel.ModelInput()
{
    CRIM = 0.02731F,
    ZN = 0F,
    INDUS = 7.07F,
    CHAS = 0F,
    NOX = 0.469F,
    RM = 6.421F,
    AGE = 78.9F,
    DIS = 4.9671F,
    RAD = 2F,
    TAX = 242F,
    PTRATIO = 17.8F,
};

//Load model and predict output
var result = ValuePredictionModel.Predict(sampleData);
Console.WriteLine($"Predicted: {result.Score}");
```

## Model Builder generated 4 files, ValuePredictionModel.consumption.cs, *.evaluate.cs *.training.cs and *.mlnet

* ValuePredictionModel.consumption.cs
  1. ModelOutput class - has the Features and Score properties that are specific to the model’s prediction results.
  1. The predicted result is assigned to the Score property of the ModelOutput class, contrary to DataClassificationModel.consumption.cs, where a target column (label) was used.

* ValuePredictionModel.training.cs
  * using Microsoft.ML.Trainers.FastTree - used by ML.NET to implement the FastTree algorithm.
  * BuildPipeline method 
    1. The ReplaceMissingValues method receives an InputOutputColumnPair array.
      1. This array indicates how each model’s input columns map to the model’s output columns.
      2. In this case, the input and output columns have the same names.
      3. So, invoking InputOutputColumnPair(@"CRIM", @"CRIM") indicates that the CRIM input column maps to the CRIM output column, and so on.
      1. In short, the ReplaceMissingValues method prepares the input data and ensures there aren’t missing values for processing.
      2. You can specify how to replace missing values, such as using the mean value in a column.
      3. The demo code uses the default technique based on the column's data type. In any event, the Boston dataset has no missing values.
    1. Following that, a series of chained calls to the Append method that add more steps to the pipeline.
    1. 1st Append -  concatenate the various input columns(except the MEDV column , which is the value the model to predict) into a new output column (Features).
     1. 2nd Append - the FastForest algorithm to use for training the model with the set of configuration options
