## Create Regression/Value Prediction Model
* Example of predicting salary based on years of experience
```csharp
 public class InputModel
 {
     public float YearsOfExperience { get; set; }
     public float Salary { get; set; }
 }
```
```csharp
  public class ResultModel
 {
     [ColumnName("Score")]
     public float Salary { get; set; }
 }
```
* var mlContext = new MLContext(seed: 1);
  1. In ML, certain algorithms and operations rely on randomness (e.g., initializing weights in NNs or splitting data into training and test sets).
  2. By specifying a seed, the same random sequence is generated every time, making the results consistent across runs.
  3. By using seed: 1, every time you run the code, you'll get the same training/test split and model performance, making the results predictable and easier to analyze.
  4. Use for Reproducibility
  5. Particularly useful during development, debugging, or when reporting results.
    
* var mlContext = new MLContext();
  1. No Seed, random number generator will use a different random seed each time the code is executed.
  1. As a result, any randomized process (e.g., shuffling data, splitting datasets) will produce non-deterministic results on each run.
  1. This is ideal for training models in real-world scenarios where you don't want the results to depend on a specific seed.

 * Recommendation in Practice
   1. During development and testing, start with a fixed seed (new MLContext(seed: 1)).
   1. Pipeline is finalized and moving towards production, switch to non-deterministic behavior (new MLContext()) for more realistic results, unless reproducibility is still a requirement.

  * ML.Net Algorithm
    1. SDCA (Stochastic Dual Coordinate Ascent) - used for training models in regression and classification tasks.
    2. FastForest (Fast Forest) - an implementation of a random forest model used for regression and classification tasks.
        1. It is part of the ensemble learning methods that combine predictions from multiple decision trees to achieve better performance.
    4. OnlineGradientDescent - is an implementation of the Stochastic Gradient Descent (SGD) optimization algorithm, used primarily for linear regression tasks.
        1. It trains models by iteratively updating weights based on the gradient of the loss function with respect to the model's parameters.
    6. LbfgsPoissonRegression - uses Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) optimization to train a Poisson regression model.
        1. It is specifically designed for regression tasks where the target variable represents count data or follows a Poisson distribution (e.g., predicting the number of events that occur in a fixed interval).
    8. Gam (Generalized Additive Models) - used for regression and binary classification tasks.
        1. GAMs are interpretable models that strike a balance between simplicity and flexibility, making them ideal for understanding the relationship between input features and the target variable.
    10. FastTree - an implementation of the gradient-boosted decision tree (GBDT) algorithm.
        1. It is used for regression, binary classification, and ranking tasks. The algorithm builds an ensemble of decision trees by iteratively improving the model’s predictions, focusing on areas where the current model performs poorly.
    12. FastTreeTweedie - a variation of the FastTree algorithm specifically designed for regression tasks where the target variable is continuous and exhibits a Tweedie distribution.
        1. This type of distribution is particularly useful for modeling data with mixed characteristics, such as zero-inflated continuous data, which is common in insurance or actuarial domains.

### 1. Create a Model
```csharp
var mlContext = new MLContext();
var data = new List<InputModel>()
{
    new InputModel { YearsOfExperience = 1, Salary = 39000 }, new InputModel { YearsOfExperience = 1.3F, Salary = 46200 },  new InputModel { YearsOfExperience = 1.5F, Salary = 37700 },
    new InputModel { YearsOfExperience = 2, Salary = 43500 }, new InputModel { YearsOfExperience = 2.2f, Salary = 40000 },   new InputModel { YearsOfExperience = 2.9f, Salary = 56000 },
    new InputModel { YearsOfExperience = 3, Salary = 60000 }, new InputModel { YearsOfExperience = 3.3f, Salary = 64000 },  new InputModel { YearsOfExperience = 3.7f, Salary = 57000 },
    new InputModel { YearsOfExperience = 3.9f, Salary = 63000 }, new InputModel { YearsOfExperience = 4, Salary = 55000 },  new InputModel { YearsOfExperience = 4, Salary = 58000 },
    new InputModel { YearsOfExperience = 4.1f, Salary = 57000 }, new InputModel { YearsOfExperience = 4.5f, Salary = 61000 },  new InputModel { YearsOfExperience = 4.9f, Salary = 68000 },
    new InputModel { YearsOfExperience = 5.3f, Salary = 83000 }, new InputModel { YearsOfExperience = 5.9f, Salary = 82000 }, new InputModel { YearsOfExperience = 6, Salary = 94000 },
    new InputModel { YearsOfExperience = 6.8f, Salary = 91000 }, new InputModel { YearsOfExperience = 7.1f, Salary = 98000 }, new InputModel { YearsOfExperience = 7.9f, Salary = 101000 },
    new InputModel { YearsOfExperience = 8.2f, Salary = 114000 }, new InputModel { YearsOfExperience = 8.9f, Salary = 109000 }
};

// Load and preprocess data
var trainingData = mlContext.Data.LoadFromEnumerable(data);

// Train a model
var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "YearsOfExperience" })
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

//the output of the pipeline after being fit to the training data. It contains all the learned parameters and transformations required for prediction.
var model = pipeline.Fit(trainingData);
```

### Save a model
* Code
  1. Save - Used to save a trained ML model to a file, it includes all transformations, trainers, and parameters needed to make predictions.
  1. Schema - The schema of the training data, which defines the structure of the data (e.g., column names, types).
  1. .zip (.mlnet) - the model is serialized and stored as a .zip file, which can later be loaded back into an ML.NET application.
* Purpose
  1. The saved model can be reused in other applications or processes without retraining.
  1. Once the model is trained and saved, it can be quickly loaded for predictions without incurring the cost of retraining.
* When to Use
  1. After training a ML model, you save it for: Deployment in prod envs, Sharing with other teams or systems, Avoiding retraining when predictions need to be made multiple times.
```csharp
// 1.Create a Model code
mlContext.Model.Save(model, trainingData.Schema, "SalaryPredictionModel.zip");
```

* Example Usage
```csharp
DataViewSchema modelSchema;
ITransformer loadedModel = mlContext.Model.Load("SalaryPredictionModel.zip", out modelSchema);

// Use loadedModel to make predictions
var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, PredictionResult>(loadedModel);
var result = predictionEngine.Predict(new InputData { YearsExperience = 5 });

Console.WriteLine($"Predicted Salary: {result.PredictedSalary}");
```



### 2. Make a prediction
```csharp
// 1.Create a Model code
var predictionEngine = mlContext.Model.CreatePredictionEngine<InputModel, ResultModel>(model);
var experience = new InputModel { YearsOfExperience = 8 };
var result = predictionEngine.Predict(experience);
Console.WriteLine($"Predicted salary for {experience.YearsOfExperience} years of experience: {result.Salary}");
```

* 3.Evaluate the model - with test data
```csharp
// 1.Create a Model code
var testData = new List<InputModel> {
    new InputModel { YearsOfExperience = 1.3F, Salary = 46200 },
    new InputModel { YearsOfExperience = 2.9F, Salary = 56000 },
    new InputModel { YearsOfExperience = 3.2F, Salary = 54000 },
    new InputModel { YearsOfExperience = 3.9F, Salary = 63000 },
    new InputModel { YearsOfExperience = 4.1F, Salary = 57000 },
    new InputModel { YearsOfExperience = 7.1F, Salary = 98000 }
};

var testDataview = mlContext.Data.LoadFromEnumerable(testData);

var metrics = mlContext.Regression.Evaluate(model.Transform(testDataview), labelColumnName: "Salary");

Console.WriteLine(
    $"R^2: {metrics.RSquared:0.00}, " +
    $"MA Error : {metrics.MeanAbsoluteError:0.00}, " +
    $"MS Error : {metrics.MeanSquaredError:0.00}, " +
    $"RMS Error : {metrics.RootMeanSquaredError:0.00}, " +
    $"Loss Function : {metrics.LossFunction:0.00}"
    );
```

* 4.Split data
   1. A value of 0.2 means 20% of the data will go to the test set, and the remaining 80% will be used as the training set.
   1. split.TrainSet: The training data (80% in this case).
   1. split.TestSet: The test data(20 % in this case).
   1. Purpose of Splitting the Dataset, is to evaluate your model's ability to generalize. By keeping part of the data unseen during training (the test set), you can check how well the model performs on new data.
 ```csharp 
 var split = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);
 var model = pipeline.Fit(split.TrainSet);
 var metrics = mlContext.Regression.Evaluate(model.Transform(split.TestSet), labelColumnName: "Salary");

 Console.WriteLine(
     $"R^2: {metrics.RSquared:0.00}, " +
     $"MA Error : {metrics.MeanAbsoluteError:0.00}, " +
     $"MS Error : {metrics.MeanSquaredError:0.00}, " +
     $"RMS Error : {metrics.RootMeanSquaredError:0.00}, " +
     $"Loss Function : {metrics.LossFunction:0.00}"
     );
```
