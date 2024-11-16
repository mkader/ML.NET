## Train a machine learning model for predictive maintenance by using ML.NET Model Builder
* https://learn.microsoft.com/en-us/training/modules/predictive-maintenance-model-builder

* Manufacturing company that uses industrial devices and equipment as part of its operations. When one of these devices breaks, it costs your company time and money. That's why performing maintenance on these devices is important.
* Scenario: Predictive maintenance
  1. There are many different factors, like usage, that affect the need for maintenance.
  2. No one device is the same.
  3. Being proactive with maintenance can help minimize the time and money that your company spends when a device breaks.
  4. Up to this point, you've been manually keeping track of which devices require maintenance.
  5. As your company expands, this process becomes more difficult to manage.
* What if you could automate predicting when a device is going to need maintenance by using sensor data?
  1. ML can help you analyze historical data from these sensors.
  2. ML can also involve learning patterns to help you predict whether a machine needs maintenance or not.
 
## What is Model Builder?
* ML is a technique that uses mathematics and statistics to identify patterns in data without being explicitly programmed.
* Model Builder is a graphical VSo extension to train and deploy custom ML models by using ML.NET.
* ![image](https://github.com/user-attachments/assets/d0cf6340-785a-4144-8d2b-a2efce5919af)
* With ML, instead of explicitly programming rules, use historical data to identify these rules based on actual observations. The patterns found through ML are then used to create an artifact called a model to make predictions by using new and previously unseen data.

* ML.NET is an open-source, cross-platform ML framework for .NET. 

* What types of problems can I solve by using Model Builder? Use Model Builder to solve many common ML problems, such as:
  1. Categorizing data: Organize news articles by topic.
  1. Predicting a numerical value: Estimate the price of a home.
  1. Grouping items with similar characteristics: Segment customers.
  1. Recommending items: Recommend movies.
  1. Classifying images: Tag an image based on its contents.
  1. Detecting objects in an image: Detect pedestrians and bicycles at an intersection.

* How can I build models by using Model Builder? Generally, the process of adding ML models to your applications consists of two steps: training and consumption.
* Training
  1. Training is the process of applying algorithms to historical data to create a model that captures underlying patterns. Use the model to make predictions on new data.
  1. Model Builder uses automated ML (AutoML) to find the best model for your data.
    1. AutoML automates the process of applying ML to data.
    1. You can run an AutoML experiment on a dataset to iterate over different data transformations, ML algorithms, and settings, and then select the best model.
  1. The model training process consists of the following steps:
    1. Choose a scenario: What problem are you trying to solve? The scenario that you choose depends on your data and what you're trying to predict.
    1. Choose an environment: Where do you want to train your model? Depending on available compute resources, cost, privacy requirements, and other factors, you might choose to train models locally on your computer or in the cloud.
    1. Load your data: Load the dataset to use for training. Define the columns that you want to predict, and then choose the columns that you want to use as inputs for your prediction.
    1. Train your model: Let AutoML choose the best algorithm for your dataset based on the scenario you've chosen.
    1. Evaluate your model: Use metrics to evaluate how well your model performs and makes predictions on new data.

*  Consumption
  1. Consumption is the process of using a trained ML model to make predictions on new and previously unseen data.
  2. With Model Builder, you can consume machine learning models from new and existing .NET projects.

* ML.NET-based ML models are serialized and saved to a file. The model file can then be loaded into any .NET application and used to make predictions through ML.NET APIs. These application types include:
  1. ASP.NET Core Web API
  1. Azure Functions
  1. Blazor
  1. Windows Presentation Foundation (WPF) or Windows Forms (WinForms)
  1. Console
  1. Class library
