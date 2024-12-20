## Adding a Model
* POCML Project ->  Right Click Add "Machine Learning Model" option -> select "Machine Learning Model (ML.NET)" and name it "DataClassificationModel.mbconfig" -> click Add.
* DataClassificationModel.mbconfig file is added, and a new Modeol Builder UI is displayed
* ![image](https://github.com/user-attachments/assets/c8b98412-641c-4bdb-93ae-3c0c9a4e4458)
* ![image](https://github.com/user-attachments/assets/6bb0859c-b258-4934-aab6-88d7edfe7910)

## Scenarios and tasks
* Before build the model and select the scenario.
* A scenario is how Model Builder describes the type of prediction that you want to make with your data, which correlates to a ML task. The task is the type of prediction based on the question being asked.
* Think of the scenario as a wrapper around a task; the task specifies the prediction type and uses various trainers (algorithms) to train the model.
* ![image](https://github.com/user-attachments/assets/4f450be7-5c77-407e-9820-c3e4c90f29f4)
* Use <a href="https://github.com/ed-freitas/ML.NET_Succinctly/blob/main/spam_assassin_tiny.csv">the dataset</a> to predict whether a text is a spam message (or not) using binary classification.
* The binary classification doesn’t appear as an option within the Model Builder UI, understand how the scenarios listed in the Model Builder UI relate to different ML tasks, such as binary classification. The binary classification task corresponds to the data classification scenario
* The Relationships between ML Tasks and Model Builder Scenarios
* ![image](https://github.com/user-attachments/assets/77aab080-da65-4de7-8a9b-9244e5932fe9)
* Binary classification is used to understand if something is positive or negative, if an email is a spam message or not, or in general, whether a particular item has a specific trait or property (or not).

## Training with Model Builder
* Choose "Data classification" the scenario -> "Select training environment" screen -> Select "Local (CPU)" 
  1. ![image](https://github.com/user-attachments/assets/be007fd0-5b0c-48d0-9275-0095ada0e0de)
* Click Next, Add data to the model -> Import "spam_assassin_tiny.csv" file -> Column to predict (Label) is set to target (the column text indicates whether the text is spam (1) or not (0)).
  1. Addd data from a local file or connect to a SQL Server database.
  1. ML.NET will use this column to predict the value based on what is read from the text column of the dataset
  1. ![image](https://github.com/user-attachments/assets/15fa0328-e84c-41fe-af8c-cc68baf5b9ee)
  1. Clicking Advanced data options.
      1. Model Builder has identified that the dataset contains two columns, text and target, as seen within Advanced data options, Column settings.
      2. The text column trains the model—it contains the actual email messages.
      3. In contrast, the target column contains the value (1 or 0) that is the value to predict.
      1. ![image](https://github.com/user-attachments/assets/7bfad88d-78d3-42ab-9a68-616661d48857)
* Click "Next Step" -> Click "Start Training", train the model with the dataset.
  1. ![image](https://github.com/user-attachments/assets/06d5def8-a3fe-479b-a3ea-d1ca8d179198)
  1. The time required to train the model is, in most cases, directly proportional to the size of the dataset.
  2. Larger dataset, the more computing resources and time are required.
  3. Typically, time is available; however, computing resources are mostly limited to the specs of the environment used.
  1. Instead of using <a href="https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset">the complete dataset</a>, which includes 5,329 rows, <a href="https://github.com/ed-freitas/ML.NET_Succinctly/blob/main/spam_assassin_tiny.csv">I created a tiny subset with only the first 100 rows (99, given that the first row is a header)</a>.
  1. While the training takes place, the different trainers (algorithms) available for the ML task will be used—highlighted.
      1. ![image](https://github.com/user-attachments/assets/10336f02-1740-452e-9570-6e0e440ffe02)
  1. Advanced training options
      1. List of the trainers that are available and used. By default, all the trainers are selected.
      2. ![image](https://github.com/user-attachments/assets/e43a3b32-7bec-45f9-9dc0-a2e098c2d5fe)
      3. It is also possible to use fewer trainers, which can be done by unchecking one or more.
      4. ML.NET has good documentation that dives deeper into what these algorithms do and how to choose one by clicking on the When should I use each algorithm option.
At this stage, I wanted to show you that it is possible to change (enable or disable) some of the predefined algorithms for training a model in case the evaluation results are not optimal.
      1. ![image](https://github.com/user-attachments/assets/6b3f2d42-45c7-4083-8c53-b5e5ed041c45)
      
## Evaluating with Model Builder
* Click Next Step -> the Evaluate option.
    1. From the spam_assassin_tiny.csv file, copy one of the rows (copied 6th row, without the target column value) and paste it into the text field above the Predict button.
    1. After clicking Predict, see how the model predicts based on the text input.
        1. The prediction is correct given that in this case the text input is not spam, but a legitimate message.
        2. Thus, the result is 0 (not spam) with a value of 67% certainty. The percentage is not a true mathematical probability, sometimes called a pseudo-probability.
    1. Feel free to try with other text input from the complete dataset.
        1. Remember that the evaluation process is an opportunity to tweak and improve the model if the results are not as expected.
        2. Trained this model with a tiny dataset, not using the entire dataset, the percentages (confidence) of the results will not be as high (when correct) or low (when incorrect) as they would be if the model had been trained using the complete dataset.
        1. So, evaluate as many times as needed and feel free to retrain the model with a slightly larger dataset to improve the accuracy of the results.
    1. The main reason I chose with a small dataset, was to save time. Try training the model with the large dataset.
    1. ![image](https://github.com/user-attachments/assets/e8384a24-dda1-47ba-bb08-d8d4e48a21e8)

## Consuming the model
* Click "Next Step", is to consume (use) the model created by Model Builder within our application.
    1. ![image](https://github.com/user-attachments/assets/2a815bd8-ac3d-41d0-96fb-7d8a33aea2c3)
    1. Model Builder makes it very easy. There are 2 options: 1) copy the Code snippet or 2) use one of the available project templates, which can be added to VS solution.
    1. I used option 1, copy the code snippet and paste it into the Main method of Program.cs
    3. ![image](https://github.com/user-attachments/assets/da73cdc2-4ba3-4d8c-8c5c-1f981deb8406)

```csharp
// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using POC_ML;

var context = new MLContext();
Console.WriteLine("Hello, World!");

//Load sample data
var sampleData = new DataClassificationModel.ModelInput()
{
    Text = @"From gort44@excite.com Mon Jun 24 17:54:21 2002 Return-Path: gort44@excite.com Delivery-Date: ...tit4unow.com/pos******************",
};

//Load model and predict output
var result = DataClassificationModel.Predict(sampleData);
Console.WriteLine($"Predicted: {result.PredictedLabel}");
```

## Model Builder generated 4 files behind the scenws - DataClassificationModel.consumption.cs, *.evaluate.cs *.training.cs and *.mlnet
* ![image](https://github.com/user-attachments/assets/c7a98004-2285-44d1-a878-10d06ba9209a)
* DataClassificationModel.consumption.cs
    1. The generated code responsible for creating the prediction engine and invoking it, allowing the consumption of the model.
    1. Microsoft.ML - includes ML.NET core methods, such as the trainers (algorithms)
    2. Microsoft.ML.Data - contains ML.NET methods that interact with the dataset used by the model.
    1. DataClassificationModel class - a partial class, declared in both DataClassificationModel.consumption.cs and DataClassificationModel.training.cs file.
    1. ModelInput class -  used as the model’s input.
    1. ModelOutput class is used as the model’s output.
        1. It contains Text, Target, Features, PredictedLabel (the predicted value) and Score (the confidence obtained for the results) properties.
        1. The Target property is an unsigned integer (uint), but the ModelInput class Target property is a string—this is because the Target property for a binary classification has to be an integer.
    1. MLNetModelPath variable - the model's metadata path (file name - DataClassificationModel.mlnet (?.zip)). The file includes the model’s schema, training information, and transformer chain metadata.
    1. PredictEngine variable - will hold the reference to the prediction engine (PredictionEngine) to make predictions on the trained model.
        1. The first parameter a lambda function that creates the engine, () => CreatePredictEngine(), and the second parameter (true) indicates whether the instance can be used by multiple threads (thread-safe).
    1. Predict method - making predictions based on the model
    1. CreatePredictEngine method - creates the prediction engine instance. out var _ parameter represents the modelInputSchema. 

* Generated model (DataClassificationModel.training.cs)
    1. It describes how the ML pipeline for the model works and behaves by specifying the various types of transformers and algorithms used and their sequence.
    1. Microsoft.ML.Trainers.FastTree library - contains the algorithm implementation used by the model.
    1. RetrainModel method - retraining the model once the pipeline has been built.
        1. It returns <a href="https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.iestimator-1?view=ml-dotnet">ITransformer</a>  - responsible for transforming data within an ML.NET model pipeline.
        1. RetrainPipeline method - building the model’s pipeline, in which the different transforms and algorithm(s) that will be used are specified.
    1. BuildPipeline method - built through a series of transformations that get subsequently added using mlContext.Transforms. 
        1. Text.<a href="https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.textcatalog.featurizetext?view=ml-dotnet">FeaturizeText</a>(inputColumnName:@"text",outputColumnName:@"text")
            1. The method transforms the input column strings (text) into numerical feature vectors (integers) that keep normalized counts of words and character n-grams.
        1. Then, a series of Append methods are chained to FeaturizeText.
        2. For every Append method, a transform operation or trainer is passed as a parameter, creating the ML.NET pipeline.
        3. The first Append, .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"text"}) - concatenate the various input columns into a new output column (Features).
        1. The next Append, .Append(mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName:@"target",inputColumnName:@"target")
            1. <a href="https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.conversionsextensionscatalog.mapvaluetokey?view=ml-dotnet">MapValueToKey</a> method - maps the input column (inputColumnName) to the output columns (outputColumnName) to convert <a href="https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/prepare-data-ml-net">categorical values</a> into keys.
        1. the next Append, this is where the magic happens, .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
           binaryEstimator:mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options()
         { NumberOfLeaves=4,MinimumExampleCountPerLeaf=20,NumberOfTrees=4, MaximumBinCountPerFeature=254,FeatureFraction=1,LearningRate=0.1,
             LabelColumnName=@"target",FeatureColumnName=@"Features"
           }),labelColumnName: @"target"))      
            1. OneVersusAll method - receives a binary estimator (binaryEstimator) algorithm instance as a parameter.
            2. The one-versus-all technique is a general ML algorithm that adapts a binary classification algorithm to handle a multiclass classification problem.
            3. The binary estimator instance - represents the ML binary classification task employed by ML.NET that contains the trainers, utilities, and options used by the <a href="https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttree?view=ml-dotnet">FastTree algorithms</a> used for making predictions on the model.
            1. Those <a href="https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttree.fasttreebinarytrainer.options?view=ml-dotnet">options</a> (NumberOfLeaves...) are then passed to the FastTree algorithms, predicting a target using a decision tree for binary classification.    
            1. The final part (labelColumnName: @"target") indicates that all predictions done by FastTree will be set on the column with the target label.

            return pipeline;
        }
    }
 }

```
