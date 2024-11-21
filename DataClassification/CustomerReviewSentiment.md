* https://dotnet.microsoft.com/en-us/apps/ai/ml-dotnet/model-builder
* Data classification - Sentiment Labelled Sentences dataset Example
* Use <a href="sentiment labelled sentences">the Sentiment Labelled Sentences datasets</a> from the <a href="https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences">UCI Machine Learning Repository</a>.
  1. Unzip and save the yelp_labelled.txt file.
  1. Each row in yelp_labelled.txt represents a different review of a restaurant left by a user on Yelp.
  2. The 1st column represents the comment left by the user, and the 2nd column represents the sentiment of the text (0 is negative, 1 is positive).
  3. The columns are separated by tabs, and the dataset has no header. The data looks like the following:
  ```
  yelp_labelled.txt
  Wow... Loved this place.	        1
  Crust is not good.	        0
  Not tasty and the texture was just nasty.	        0
  ```

* Add data -> dataset does not have a header, headers are auto-generated ("col0" and "col1") -> Under Column to predict (Label), select "col1", which is the sentiment found in "col1".
  1. The columns that are used to help predict the Label are called Features. All of the columns (col0) in the dataset besides the Label are automatically selected as Features.
  1. ![image](https://github.com/user-attachments/assets/881c4b4d-db7c-4601-af21-517814677c70)

* Train your model
  1. Model Builder evaluates many models with varying algorithms and settings based on the amount of training time given to build the best performing model.
  1. Change the Time to train (default 10 secs), which is the amount of time Model Builder to explore various models, to 60 seconds (try increasing this number if no models are found after training) .
  1. Note that for larger datasets, the training time will be longer. Model Builder automatically adjusts the training time based on the dataset size.
  1. You can update the optimization metric and algorithms used in Advanced training options, but it is not necessary for this example.
  1. ![image](https://github.com/user-attachments/assets/047f7921-8689-437f-bfb8-4ad0f215dd10)

* Training Results
  1. Best MacroAccuracy - the accuracy of the best model that Model Builder found. Higher accuracy means the model predicted more correctly on test data.
  1. Best model - algorithm performed the best during Model Builder's exploration.
  1. Training time - the total amount of time that was spent training / exploring models.
  1. Models explored (total) - the total number of models explored by Model Builder in the given amount of time.
  1. Generated code-behind - the names of the files generated to help consume the model or train a new model.
  1. If you want, you can view more information about the training session in the Machine Learning Output window.
  1. ![image](https://github.com/user-attachments/assets/2ab0e544-e838-450c-b7fb-27c02124ab88)
  ```
  start multiclass classification
  Evaluate Metric: MacroAccuracy
  Available Trainers: LGBM,FASTFOREST,FASTTREE,LBFGS,SDCA
  Training time in second: 60
  |      Trainer                             MacroAccuracy Duration    |
  |--------------------------------------------------------------------|
  |0     FastTreeOva                         0.6643     2.9640         |
  |1     FastForestOva                       0.7180     2.9970         |
  |2     FastTreeOva                         0.7485     2.4340         |
  |3     LightGbmMulti                       0.6823     2.0310         |
  |4     SdcaMaximumEntropyMulti             0.5000     1.3820         |
  |5     FastTreeOva                         0.7376     3.2370         |
  |6     FastTreeOva                         0.7031     2.2470         |
  |7     LbfgsMaximumEntropyMulti            0.7076     1.3540         |
  |8     FastForestOva                       0.7467     5.9190         |
  [Source=AutoMLExperiment, Kind=Info] cancel training because cancellation token is invoked...
  |--------------------------------------------------------------------|
  |                          Experiment Results                        |
  |--------------------------------------------------------------------|
  |                               Summary                              |
  |--------------------------------------------------------------------|
  |ML Task: multiclass classification                                  |
  |Dataset: C:\sentiment labelled sentences\yelp_labelled.txt|
  |Label : col1                                                        |
  |Total experiment time :    59.0000 Secs                             |
  |Total number of models explored: 10                                 |
  |--------------------------------------------------------------------|
  |                        Top 5 models explored                       |
  |--------------------------------------------------------------------|
  |      Trainer                             MacroAccuracy Duration    |
  |--------------------------------------------------------------------|
  |2     FastTreeOva                         0.7485     2.4340         |
  |8     FastForestOva                       0.7467     5.9190         |
  |5     FastTreeOva                         0.7376     3.2370         |
  |1     FastForestOva                       0.7180     2.9970         |
  |7     LbfgsMaximumEntropyMulti            0.7076     1.3540         |
  |--------------------------------------------------------------------|
  Generate code behind files
  
  
  Copying generated code to project...
  Copying RestaurantReviewModel.consumption.cs to folder: C:\src\POC.ML
  Copying RestaurantReviewModel.training.cs to folder: C:\src\POC.ML
  Copying RestaurantReviewModel.evaluate.cs to folder: C:\src\POC.ML
  COMPLETED

  Updating nuget dependencies...
  Starting update NuGet dependencies async.
  Project Miscellaneous Files cannot be accessed. It may be unloaded.
  Installing nuget package, package ID: Microsoft.ML, package Version: 2.0.0
  Installing nuget package, package ID: Microsoft.ML.FastTree, package Version: 2.0.0
  COMPLETED
  ```

* Evaluate your model
  1. Make predictions on sample input in the Try your model section.
  2. The textbox is pre-filled with the first line of data from your dataset, but you can change the input and select the Predict button to try out different sentiment predictions.
  1. In this case, 0 means negative sentiment and 1 means positive sentiment.
  1. If your model is not performing well (for example, if the Accuracy is low or if the model only predicts '1' values), you can try adding more time and training again.
  1. This is a sample using a very small dataset; for production-level models, you'd want to add a lot more data and training time.

* Generate code
  1. After training is completed, four files are automatically added as code-behind to the SentimentModel.mbconfig:
    1. SentimentModel.consumption.cs - contains the model input and output classes and a Predict method that can be used for model consumption.
    1. SentimentModel.evaluate.cs - contains a CalculatePFI method that uses the Permutation Feature Importance (PFI) technique to evaluate which features contribute most to the model predictions.
    1. SentimentModel.mlnet - This file is the trained ML.NET model, which is a serialized zip file.
      1. ![image](https://github.com/user-attachments/assets/dcc3911f-0d5f-4b17-8972-41005698cc18)
    1. SentimentModel.training.cs - contains the code to understand the importance input columns have on your model predictions.
    1. ![image](https://github.com/user-attachments/assets/73c481af-8f9a-417c-9b69-4adf09554d41)
  1. A code snippet is provided which creates sample input for the model and uses the model to make a prediction on that input.
  1. Model Builder also offers 2 Project templates that you can optionally add to your solution. 1. a console app and 2. a web API, both which consume the trained model.
    1.  ![image](https://github.com/user-attachments/assets/79d9cb04-d530-4b5a-8eed-0d8fc378ee44)

* Consume your model
```csharp
Program.cs
using MyMLApp;
// Add input data
var sampleData = new SentimentModel.ModelInput()
{
    Col0 = "This restaurant was wonderful."
};

// Load model and predict output of sample data
var result = SentimentModel.Predict(sampleData);

// If Prediction is 1, sentiment is "Positive"; otherwise, sentiment is "Negative"
var sentiment = result.PredictedLabel == 1 ? "Positive" : "Negative";
Console.WriteLine($"Text: {sampleData.Col0}\nSentiment: {sentiment}");
```
  1. ![image](https://github.com/user-attachments/assets/8ce404b3-86c1-45a5-a166-e2bbabc0a888)


