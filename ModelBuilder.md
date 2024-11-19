* https://learn.microsoft.com/en-us/training/modules/predictive-maintenance-model-builder

## What is Model Builder?
* ML is a technique that uses mathematics and statistics to identify patterns in data without being explicitly programmed.
* Model Builder is a graphical VS extension to train and deploy custom ML models by using ML.NET.
* <img src="https://github.com/user-attachments/assets/d0cf6340-785a-4144-8d2b-a2efce5919af">

* ML.NET is an open-source, cross-platform ML framework for .NET. 

* 2 Steps to build models by using Model Builder - 1.training and 2. consumption.
  1. Training
      1. Training is the process of applying algorithms to historical data to create a model that captures underlying patterns. Use the model to make predictions on new data.
      1. Model Builder uses automated ML (AutoML) to find the best model for your data.
          1. AutoML automates the process of applying ML to data.
          1. Run an AutoML experiment on a dataset to iterate over different data transformations, ML algorithms, and settings, and then select the best model.
      1. The model training process consists of the following steps:
          1. Choose a scenario: Problem trying to solve? Choose depends on your data and trying to predict.
          1. Choose an environment: Train your model locally or cloud? Depending on available compute resources, cost, privacy requirements, and other factors
          2. Load your data: Use for training. Define the columns to predict and then choose the columns to use as inputs for prediction.
          1. Train your model: Let AutoML choose the best algorithm for your dataset based on the scenario you've chosen.
          1. Evaluate your model: Use metrics to evaluate how well your model performs and makes predictions on new data.

  1. Consumption - is the process of using a trained ML model to make predictions on new and previously unseen data.

* ML.NET-based ML models are serialized and saved to a file. It can then be loaded into any .NET application and used to make predictions through ML.NET APIs.
  1. .NET Applications: ASP.NET Core Web API, Azure Functions, Blazor,  Windows Presentation Foundation (WPF) or Windows Forms (WinForms), Console, Class library

## Choose your scenario and prepare data

### Start the training process
* Add a new ML Model (ML.NET) item to .NET application with the .mbconfig (Model Builder configuration) authored in JSON. These files allow you to:
    1. Provide a name for your model.
    1. Collaborate with others on your team via source control.
    1. Persist state. If at any point in the training process you need to close Model Builder, your state is saved and you can pick up where you left off.

|<img src="https://github.com/user-attachments/assets/d2f148e3-61cc-44f3-8bc7-1409c8ab22c7">|<img src="https://github.com/user-attachments/assets/7e706b97-3d50-4652-9c20-e5f13a835a54" >|
    
### STEP 1: Choose a scenario
* What is a scenario? - A scenario describes the problem you're trying to solve by using your data. Common scenarios:
  1. Categorizing data: Organize news articles by topic.
  1. Predicting a numerical value: Estimate the price of a home.
  1. Grouping items with similar characteristics: Segment customers.
  1. Classifying images: Tag an image based on its contents.
  1. Recommending items: Recommend movies.
  1. Detecting objects in an image: Detect pedestrians and bicycles at an intersection.
* The scenarios map to ML tasks. A ML task is the type of prediction or inference being made, based on the problem or question that's being asked and the available data.

* 2 categories ML tasks: 1.Supervised, 2.Unsupervised
  1. For supervised ML tasks, the label is known. Examples: Classification, Binary (two categories), Multiclass (two or more categories), Image, Regression
  1. For unsupervised ML tasks, the label is unknown. Examples: Clustering, Anomaly detection, Supported scenarios in Model Builder

* Model Builder supports the following scenarios that map to machine learning tasks:

| Scenario | Machine learning task | Use case |
|-|-|-|
| Data classification	| Binary and multiclass classification	| Organize articles by topic. |
| Value prediction	| Linear regression	 |Predict the price of a home based on features of the home. |
| Image classification	| Image classification (deep learning)	| Organize images by animal species based on the content of an image. |
| Recommendation	| Recommendation	| Recommend movies based on the preferences of similar users. |
| Object detection	| Object detection (deep learning)	| Identify physical damage in an image. |

* <img src="https://github.com/user-attachments/assets/1653a8e2-5269-4445-bd53-3a6eb52e8a32">
* <img src="https://github.com/user-attachments/assets/cedf7b17-2fdb-4a85-9b3f-48850a2eb0df">


### STEP 2: Choose your environment - To train ML model. Supported environments in Model Builder

| Scenario	| Local CPU	| Local GPU	| Azure GPU |
| -	| -| -	| - |
| Data classification	| ✔️	| ❌	| ❌ |
| Value prediction	| ✔️	| ❌	| ❌ |
| Image classification	| ✔️	| ✔️	| ✔️ |
| Recommendation	| ✔️	| ❌	| ❌ |
| Object detection	| ❌	| ❌	| ✔️ |

* Local environments - reasons to consider
  1. Training locally doesn't cost you anything because you're using your computer's resources.
  1. You don't want your data to leave your computer or datacenter.

* Azure environments
  1. Scenarios like image classification and object detection are resource intensive.
  2. Using a GPU can often speed up the training process.
  3. If you don't have a GPU or a computer with enough CPU or RAM, offloading the training process to Azure can lighten the load on your system.

 * ![image](https://github.com/user-attachments/assets/a9105aa7-a125-496a-95b0-23fde5c6a215)

### STEP 3: Load and prepare your data
* The process for loading data into Model Builder consists of three steps:
  1. Choose your data source type. 2. Provide the location of your data. 3. Choose column purpose.

* Choose your data source type, Model Builder supports loading data from the following sources:
  1. Delimited files (,, ;, and tab), 2. SQL Server dbs, 3. Images, 4. Provide the location (a directory, file path, or db connection string) where your dataset is stored.
  
* When a data source is selected in Model Builder, it parses the data and makes its best effort to identify:
  1. Headers and column names, 2. Column separator, 3. Column data types, 4. Column purpose, 5. Decimal separators

* After the data is loaded, Model Builder displays a preview of some of the elements in your dataset.

* Choose column purpose
  1. In scenarios like data classification and value prediction, choose which of columns is the column that you want to predict (label).
  1. By default, all other columns that are not the label are used as features. Features are columns used as inputs to predict the label.

* ![image](https://github.com/user-attachments/assets/b218a09e-5181-4cd7-927b-f8adc98d5e98)

* Advanced data options - To customize how your data is loaded. Customize settings
  1. For columns, choose the following settings:
    1. Purpose: Should the column be a feature, be a label, or be ignored? You can have only one column selected as the label.
    1. Data type: Is the value a single-precision float value, string, or Boolean?
    1. Categorical: Does the column represent a categorical value (for example: low, medium, or high)?
  1. To format data, choose whether the data contains column headers, the column separator (,, ;, or tab), and the decimal separator type (period or comma).
  2. ![image](https://github.com/user-attachments/assets/7b0e11e3-35fe-446e-a0e0-c048dd08c7c5)
  3. ![image](https://github.com/user-attachments/assets/cd210c3a-9fb6-4a4b-b237-168529fc88ca)
  4. ![image](https://github.com/user-attachments/assets/abd7f24d-2f88-4138-b254-77aaa4188949)
