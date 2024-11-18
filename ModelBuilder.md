* https://learn.microsoft.com/en-us/training/modules/predictive-maintenance-model-builder

## What is Model Builder?
* ML is a technique that uses mathematics and statistics to identify patterns in data without being explicitly programmed.
* Model Builder is a graphical VS extension to train and deploy custom ML models by using ML.NET.
* ![image](https://github.com/user-attachments/assets/d0cf6340-785a-4144-8d2b-a2efce5919af)

* ML.NET is an open-source, cross-platform ML framework for .NET. 

* Types of problems by using Model Builder?
  1. Categorizing data: Organize news articles by topic.
  1. Predicting a numerical value: Estimate the price of a home.
  1. Grouping items with similar characteristics: Segment customers.
  1. Recommending items: Recommend movies.
  1. Classifying images: Tag an image based on its contents.
  1. Detecting objects in an image: Detect pedestrians and bicycles at an intersection.

* 2 Steps to build models by using Model Builder - 1.training and 2. consumption.
* Training
  1. Training is the process of applying algorithms to historical data to create a model that captures underlying patterns. Use the model to make predictions on new data.
  1. Model Builder uses automated ML (AutoML) to find the best model for your data.
      1. AutoML automates the process of applying ML to data.
      1. Run an AutoML experiment on a dataset to iterate over different data transformations, ML algorithms, and settings, and then select the best model.
  1. The model training process consists of the following steps:
      1. Choose a scenario: What problem are you trying to solve? The scenario that you choose depends on your data and what you're trying to predict.
          1. A scenario describes the problem you're trying to solve by using your data. Common scenarios include the ones listed in the previous unit:
              1. Categorizing data: Organize news articles by topic.
              1. Predicting a numerical value: Estimate the price of a home.
              1. Grouping items with similar characteristics: Segment customers.
              1. Classifying images: Tag an image based on its contents.
              1. Recommending items: Recommend movies.
              1. Detecting objects in an image: Detect pedestrians and bicycles at an intersection.
          1. The scenarios map to ML tasks. A ML task is the type of prediction or inference being made, based on the problem or question that's being asked and the available data.
       1. Choose an environment: Where do you want to train your model? Depending on available compute resources, cost, privacy requirements, and other factors, you might choose to train models locally on your computer or in the cloud.
      1. Load your data: Load the dataset to use for training. Define the columns that you want to predict, and then choose the columns that you want to use as inputs for your prediction.
      1. Train your model: Let AutoML choose the best algorithm for your dataset based on the scenario you've chosen.
      1. Evaluate your model: Use metrics to evaluate how well your model performs and makes predictions on new data.

* Consumption
  1. Consumption is the process of using a trained ML model to make predictions on new and previously unseen data.
  2. With Model Builder, you can consume ML models from new and existing .NET projects.

* ML.NET-based ML models are serialized and saved to a file. The model file can then be loaded into any .NET application and used to make predictions through ML.NET APIs. These application types include:
  1. ASP.NET Core Web API
  1. Azure Functions
  1. Blazor
  1. Windows Presentation Foundation (WPF) or Windows Forms (WinForms)
  1. Console
  1. Class library

## Choose your scenario and prepare data
* Start the training process
  1. Add a new ML Model (ML.NET) item to a new or existing .NET application.
  1. The ML Model (ML.NET) item adds a file to your project with the .mbconfig (Model Builder configuration) file extension authored in JSON. These files allow you to:
      1. Provide a name for your model.
      1. Collaborate with others on your team via source control.
      1. Persist state. If at any point in the training process you need to close Model Builder, your state is saved and you can pick up where you left off.
  
* Machine learning tasks tend to fall into two categories: 1.Supervised, 2.Unsupervised
  1. For supervised tasks, the label is known. Examples of supervised machine learning tasks include:
      1. Classification
      1. Binary (two categories)
      1. Multiclass (two or more categories)
      1. Image
      1. Regression
  1. For unsupervised tasks, the label is unknown. Examples of unsupervised machine learning tasks include:
      1. Clustering
      1. Anomaly detection
      1. Supported scenarios in Model Builder

* Model Builder supports the following scenarios that map to machine learning tasks:

| Scenario | Machine learning task | Use case |
|-|-|-|
| Data classification	| Binary and multiclass classification	| Organize articles by topic. |
| Value prediction	| Linear regression	 |Predict the price of a home based on features of the home. |
| Image classification	| Image classification (deep learning)	| Organize images by animal species based on the content of an image. |
| Recommendation	| Recommendation	| Recommend movies based on the preferences of similar users. |
| Object detection	| Object detection (deep learning)	| Identify physical damage in an image. |

Choosing a scenario for predictive maintenance
Depending on what your data looks like, the predictive maintenance problem can be modeled through different tasks. For your use case, because the label is a binary value (0 or 1) that describes whether a machine is broken or not, the data classification scenario is appropriate.

Choose your environment
Now that you've selected the data classification, it's time to choose the environment where you want to train your machine learning model. Environments describe the compute resources that you'll use to train your machine learning model.

Supported environments in Model Builder
Model Builder supports the following environment options:

Scenario	Local CPU	Local GPU	Azure GPU
Data classification	✔️	❌	❌
Value prediction	✔️	❌	❌
Image classification	✔️	✔️	✔️
Recommendation	✔️	❌	❌
Object detection	❌	❌	✔️
Depending on your use case, there are various reasons why you might choose a local or Azure environment.

Local environments
Reasons why you might consider local compute environments include:

Training locally doesn't cost you anything because you're using your computer's resources.
You don't want your data to leave your computer or datacenter.
Azure environments
Scenarios like image classification and object detection are resource intensive. Using a GPU can often speed up the training process. If you don't have a GPU or a computer with enough CPU or RAM, offloading the training process to Azure can lighten the load on your system.

Load and prepare your data
With your scenario and training environment selected, it's time to load and prepare your data.

Data is the most important component for building machine learning models. The process for loading data into Model Builder consists of three steps:

Choose your data source type.
Provide the location of your data.
Choose column purpose.
Choose your data source type
Depending on your scenario, Model Builder supports loading data from the following sources:

Delimited files (comma, semicolon, and tab)
Local and remote SQL Server databases
Images (.jpg and .png)
Provide the location of your data
After you've selected your data source type, you have to provide the location where your dataset is stored. This location can be a directory, file path, or database connection string. It's dependent on the scenario and data source type that you've selected.

When a data source is selected in Model Builder, it parses the data and makes its best effort to identify:

Headers and column names
Column separator
Column data types
Column purpose
Decimal separators
After the data is loaded, Model Builder displays a preview of some of the elements in your dataset.

Choose column purpose
Depending on the scenario that you choose, you'll have to define the purpose of certain columns. In scenarios like data classification and value prediction, you'll have to choose which of your columns is the column that you want to predict (label).

By default, all other columns that are not the label are used as features. Features are columns used as inputs to predict the label.

Advanced data options
To customize how your data is loaded, Model Builder provides advanced data options. These options allow you to customize settings that pertain to your columns and dataset formatting.

For columns, you can choose the following settings:

Purpose: Should the column be a feature, be a label, or be ignored? You can have only one column selected as the label.
Data type: Is the value a single-precision float value, string, or Boolean?
Categorical: Does the column represent a categorical value (for example: low, medium, or high)?
To format data, you can choose whether the data contains column headers, the column separator (comma, semicolon, or tab), and the decimal separator type (period or comma).

Understanding the predictive maintenance dataset
The dataset used in this module is the AI4I 2020 Predictive Maintenance Dataset. This synthetic dataset reflects real predictive maintenance data encountered in industry. It consists of 10,000 data points and 14 columns. It's from:

Explainable Artificial Intelligence for Predictive Maintenance Applications, Stephan Matzka, Third International Conference on Artificial Intelligence for Industries (AI4I 2020), 2020 (in press), and hosted at the UCI Machine Learning Repository - Dua, D. and Graff, C. (2019). UCI Machine Learning Repository http://archive.ics.uci.edu/ml. Irvine, CA: University of California, School of Information and Computer Science.

Here's a preview of what the data looks like:

UDI	Product ID	Type	Air temperature [K]	Process temperature [K]	Rotational speed [rpm]	Torque [Nm]	Tool wear [min]	Machine failure	TWF	HDF	PWF	OSF	RNF
1	M14860	M	298.1	308.6	1551	42.8	0	0	0	0	0	0	0
162	L47341	L	298.3	308.1	1412	52.3	218	1	0	0	0	1	0
The columns are defined as follows:

UDI: The row's index.
Product ID: A product identifier that contains a product type category and a variant-specific serial number.
Type: Product quality category. The values are L (low; 50% of all products), M (medium; 30%), or H (high; 20%).
Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]: Values collected from sensors.
Machine failure: Binary label (0 or 1) that indicates whether the machine has failed.
TWF, HDF, PWF, OSF, RNF: Independent machine failure modes. A value of 1 indicates that the respective failure mode occurred.
For this scenario, you won't use all of the columns in the dataset because they either don't inform the prediction or contain redundant information.

Because you want to be able to predict whether a machine will fail or not, the Machine failure column is the label. In Model Builder, for features, you can use data from the Product ID, Type, and various sensor columns.

Although the failure modes are useful in diagnosing the root cause of a failure, they aren't useful for your use case. The reason is that you only want to know whether a machine has failed or not. Additionally, similar information is already captured by the Machine failure column. So, you can ignore these columns.

Proceed to the next unit to check your knowledge on scenarios, environments, and data preparation.

Next unit: Knowledge check - Scenario and data
