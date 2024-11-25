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
### MLContext
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

### ML.Net Algorithm
* SDCA (Stochastic Dual Coordinate Ascent) - used for training models in regression and classification tasks.
* FastForest (Fast Forest) - an implementation of a random forest model used for regression and classification tasks.
  1. It is part of the ensemble learning methods that combine predictions from multiple decision trees to achieve better performance.
* OnlineGradientDescent - is an implementation of the Stochastic Gradient Descent (SGD) optimization algorithm, used primarily for linear regression tasks.
  1. It trains models by iteratively updating weights based on the gradient of the loss function with respect to the model's parameters.
* LbfgsPoissonRegression - uses Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) optimization to train a Poisson regression model.
  1. It is specifically designed for regression tasks where the target variable represents count data or follows a Poisson distribution (e.g., predicting the number of events that occur in a fixed interval).
* Gam (Generalized Additive Models) - used for regression and binary classification tasks.
  1. GAMs are interpretable models that strike a balance between simplicity and flexibility, making them ideal for understanding the relationship between input features and the target variable.
* FastTree - an implementation of the gradient-boosted decision tree (GBDT) algorithm.
  1. It is used for regression, binary classification, and ranking tasks. The algorithm builds an ensemble of decision trees by iteratively improving the model’s predictions, focusing on areas where the current model performs poorly.
* FastTreeTweedie - a variation of the FastTree algorithm specifically designed for regression tasks where the target variable is continuous and exhibits a Tweedie distribution.
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

// Load and preprocess data. LoadFromEnumerable - Loads the data from an in-memory collection (data) into an IDataView.
var trainingData = mlContext.Data.LoadFromEnumerable(data);

// Train a model
var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "YearsOfExperience" })
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

//the output of the pipeline after being fit to the training data. It contains all the learned parameters and transformations required for prediction.
var model = pipeline.Fit(trainingData);
```

### Save and Load a model
* Code
  1. Save - Used to save a trained ML model to a file, it includes all transformations, trainers, and parameters needed to make predictions.
  1. Schema - The schema of the training data, which defines the structure of the data (e.g., column names, types).
  1. .zip (.mlnet) - the model is serialized and stored as a .zip file, which can later be loaded back into an ML.NET application.
* Purpose
  1. The saved model can be reused in other applications or processes without retraining.
  1. Once the model is trained and saved, it can be quickly loaded for predictions without incurring the cost of retraining.
* When to Use - After training a ML model, you save it for:
  1. Deployment in prod envs
  2. Sharing with other teams or systems
  3. Avoiding retraining when predictions need to be made multiple times.
```csharp
// 1.Create a Model code
//save model
mlContext.Model.Save(model, trainingData.Schema, "SalaryPredictionModel.zip");

OR 
using (FileStream stream = new FileStream(@"SalaryPredictionModel.zip", FileMode.Create)) 
{
    mlContext.Model.Save(model, trainingData.Schema, stream);
}

OR
var file = new MultiFileSource(@"combined.csv");
// new MultiFileSource("combined.csv", "combined2.csv").
var dataLoader = mlContext.Data.CreateTextLoader<InputModel>(separatorChar: ',', hasHeader: true, dataSample: file);
mlContext.Model.Save(model, dataLoader, @"SalaryPredictionModel.zip");
```

* Load model
```csharp
ITransformer loadedModel = mlContext.Model.Load("SalaryPredictionModel.zip", out DataViewSchema modelSchema);

OR

 ITransformer loadedModel;

 using (FileStream stream = new FileStream(@"SalaryPredictionModel.zip", FileMode.OpenOrCreate))
     loadedModel = mlContext.Model.Load(stream, out DataViewSchema modelSchema);

OR
mlContext.Model.LoadWithDataLoader(@"SalaryPredictionModel.zip", out IDataLoader<IMultiStreamSource> loader);

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

Console.WriteLine($"R^2: {metrics.RSquared:0.00}, MA Error : {metrics.MeanAbsoluteError:0.00}, " +
    $"MS Error : {metrics.MeanSquaredError:0.00}, RMS Error : {metrics.RootMeanSquaredError:0.00}, " +
    $"Loss Function : {metrics.LossFunction:0.00}");
```

* 4.Split data - Evaluate the model - with same dataset
   1. A value of 0.2 means 20% of the data will go to the test set, and the remaining 80% will be used as the training set.
   1. split.TrainSet: The training data (80% in this case).
   1. split.TestSet: The test data(20 % in this case).
   1. Purpose of Splitting the Dataset, is to evaluate your model's ability to generalize. By keeping part of the data unseen during training (the test set), you can check how well the model performs on new data.
 ```csharp 
 var trainingData = mlContext.Data.LoadFromEnumerable(data);

 var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "YearsOfExperience" })
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

 var split = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);
 var model = pipeline.Fit(split.TrainSet);
 var metrics = mlContext.Regression.Evaluate(model.Transform(split.TestSet), labelColumnName: "Salary");

 Console.WriteLine($"R^2: {metrics.RSquared:0.00}, MA Error : {metrics.MeanAbsoluteError:0.00}, " +
    $"MS Error : {metrics.MeanSquaredError:0.00}, RMS Error : {metrics.RootMeanSquaredError:0.00}, " +
    $"Loss Function : {metrics.LossFunction:0.00}");
```
### Cross Validate Model
* Splits the data into 5 folds (default is usually 80% training and 20% validation per fold).
* Trains and evaluates the model on each fold.
* Output - Returns a collection of results (crossValidateResult), each containing:
   1. Model: The trained model for the fold.
   1. Metrics: Evaluation metrics for that fold.
* Runs cross-validation to evaluate the pipeline on different data splits.
```csharp
 //4.Cross Validate Model
 var trainingData = mlContext.Data.LoadFromEnumerable(data);

 var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "YearsOfExperience" })
   .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

 var crossValidateResult = mlContext.Regression.CrossValidate(trainingData, pipeline, numberOfFolds: 5, labelColumnName: "Salary");

// Extracting Metrics and Models
 var allMetrics = crossValidateResult.Select(fold => fold.Metrics);
 var allModesl = crossValidateResult.Select(fold => fold.Model);

//Evaluating Each Fold
 foreach (var metrics in allMetrics)
 {
     Console.WriteLine($"R^2: {metrics.RSquared:0.00}, MA Error : {metrics.MeanAbsoluteError:0.00}, " +
       $"MS Error : {metrics.MeanSquaredError:0.00}, RMS Error : {metrics.RootMeanSquaredError:0.00}, " +
       $"Loss Function : {metrics.LossFunction:0.00}");
 }

// Finding the Best-Performing Model

// Sorts all metrics by  model.RSquared(higher is better) and picks the fold with the highest model.RSquared
 var bestPerformance = allMetrics.OrderByDescending(model => model.RSquared).First();

//Finds the index of the best metrics and retrieves the corresponding model from allModels.
 var bestPerformanceIndex = allMetrics.ToList().IndexOf(bestPerformance);
 var bestModel = allModesl.ElementAt(bestPerformanceIndex);

 Console.WriteLine("Average Data");

 Console.WriteLine($"R^2: {allMetrics.Average(x => x.RSquared):0.00}," +
     $" MA Error : {allMetrics.Average(x => x.MeanAbsoluteError):0.00}, " +
       $"MS Error : {allMetrics.Average(x => x.MeanSquaredError):0.00}, " +
       $"RMS Error : {allMetrics.Average(x => x.RootMeanSquaredError):0.00}, " +
       $"Loss Function : {allMetrics.Average(x => x.LossFunction):0.00}");
```
![image](https://github.com/user-attachments/assets/2dde6c50-faaf-4a98-a9a6-8e23c20c8116)

### Algorithms & Hyperparameters
```csharp
var trainingData = mlContext.Data.LoadFromEnumerable(data);

var estimator = mlContext.Transforms.Concatenate("Features", new[] { "YearsOfExperience" });

// SDCA Regression (100 iterations) - Achieves R²: 0.91, which indicates a strong fit.
// Performs well, as Stochastic Dual Coordinate Ascent (SDCA) is effective for regression tasks.
var pipeline = estimator.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100)); //output => R^2: 0.91

// SDCA Regression (10 iterations) - Results in R²: 0.52, showing underfitting due to fewer iterations.
var pipeline = estimator.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 10)); //output => R^2: 0.52

// SDCA with L1/L2 Regularization - Produces R²: 0.78, suggesting slight underfitting due to regularization penalties.
// Adding L1 and L2 regularization helps prevent overfitting but might reduce performance slightly if the data doesn't require regularization.
var pipeline = estimator.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100, l1Regularization:2, l2Regularization:2)); //output => R^2: 0.78

// LBFGS Poisson Regression - Delivers R²: 0.96, the best result, indicating excellent fit to the data.
// This method is highly accurate for the given data. LBFGS is a robust optimizer, and Poisson regression is suitable for certain types of regression problems.
var pipeline = estimator.Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Salary")); //output => R^2: 0.96

// LBFGS with High Optimization Tolerance - Produces R²: -9.91, a poor result due to inappropriate optimization parameters.
// Poor performance likely due to an overly high optimization tolerance, causing the model to diverge.
var pipeline = estimator.Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Salary", optimizationTolerance:100)); //output => R^2: -9.91

// Online Gradient Descent (default) - Results in R²: 0.27, showing significant underfitting.
// This online algorithm updates weights iteratively but is less effective for this dataset. It underfits the data.
var pipeline = estimator.Append(mlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Salary")); //output => R^2: 0.27

// Online Gradient Descent (100 iterations) - Achieves R²: 0.91, comparable to SDCA with 100 iterations.
// Increasing iterations improves model accuracy significantly, matching SDCA's performance.
var pipeline = estimator.Append(mlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Salary", numberOfIterations:100)); //output => R^2: 0.91

var model = pipeline.Fit(trainingData);

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

Console.WriteLine($"R^2: {metrics.RSquared:0.00}");
```

 ### Load data from Datanase and  Text and binary File (single, multi files - same or different folder)
 * single file - csv
 ```chsarp
// LoadFileInputModel and InputModel - difference is LoadColumn attribute
var dataView = mlContext.Data.LoadFromTextFile<LoadFileInputModel>(path: @"Salary\train-dataset.csv", separatorChar: ',', hasHeader: true);
```

* single file - tsv (tab delimited), no header in file, so hasHeader is false or removed.
```chsarp
var dataView = mlContext.Data.LoadFromTextFile<LoadFileInputModel>(path: @"Salary\train-dataset.tsv", hasHeader: true);
```
* single file - without model.
```chsarp
var columnsToLoad = new TextLoader.Column[]
{
    new TextLoader.Column("YearsOfExperience", DataKind.Single, 0),
    new TextLoader.Column("Salary", DataKind.Single, 1)
};

var dataView = mlContext.Data.LoadFromTextFile(path: @"Salary\train-dataset.csv",  separatorChar: ',', hasHeader: true, columns: columnsToLoad);
```

* multi files - same folder
```chsarp
var dataView = mlContext.Data.LoadFromTextFile<LoadFileInputModel>(path: @"Salary\train-datasets\*", separatorChar: ',', hasHeader: false);
```

* multi files - different folder
```chsarp
var textLoader = mlContext.Data.CreateTextLoader<LoadFileInputModel>(separatorChar: ',');
var dataView = textLoader.Load(@"Salary\train-datasets\train-dataset1.csv", @"Salary\train-datasets\train-dataset2.csv");
```

* single file - binary
```chsarp
var dataView = mlContext.Data.LoadFromBinary"(Salary\train-dataset.bin");
```

* load from Database
```chsarp
 var dbLoader = mlContext.Data.CreateDatabaseLoader<InputModel>();

 var conStr = @"Server=.;Database=mlnet;Integrated Security=True;TrustServerCertificate=True;";
 var qry = @"SELECT YearsOfExperience, Salary FROM [mlnet].[dbo].[SalaryInfo]";
 var dbSource = new DatabaseSource(SqlClientFactory.Instance, conStr, qry);

 IDataView dataView = dbLoader.Load(dbSource);
```

* Development purpose, not for production. Default is 100 rows, use maxRows to load all rows
```chsarp
var preview = dataView.Preview();
```
![image](https://github.com/user-attachments/assets/ef5097ba-ce27-4b0f-8e67-1424d1732fee)

 ### Save data - csv, tsv, binary
 ```chsarp
 var textLoader = mlContext.Data.CreateTextLoader<InputModel>(separatorChar: ',');
 var dataView = textLoader.Load(@train_datasets\1.csv", @"C:\ml_net\POC.MLNET\train_datasets\2.csv");

 var list = mlContext.Data.CreateEnumerable<InputModel>(dataView, false).ToList();

 using (FileStream stream = new FileStream(@"train_datasets\combined.tsv", FileMode.OpenOrCreate))
 {
     mlContext.Data.SaveAsText(dataView, stream);
 }

 using (FileStream stream = new FileStream(@"C:\ml_net\POC.MLNET\train_datasets\combined.csv", FileMode.OpenOrCreate))
 {
     mlContext.Data.SaveAsText(dataView, stream, separatorChar:',', headerRow:false, schema: false);
 }

 using (FileStream stream = new FileStream(@"C:\ml_net\POC.MLNET\train_datasets\combined.bin", FileMode.OpenOrCreate))
 {
     mlContext.Data.SaveAsBinary(dataView, stream);
 }
```
![image](https://github.com/user-attachments/assets/be60a649-5e9a-403d-b62f-fd8f3dd36f99)

### ShuffleRow, Skipped Data, Take Date and Filter
 ```chsarp
//shuffled data
var shuffledData = mlContext.Data.ShuffleRows(dataView);
preview = shuffledData.Preview();

//skip first 8 records
var skippedData = mlContext.Data.SkipRows(dataView, 8);
preview = skippedData.Preview();

//take first 8 records
var takeData = mlContext.Data.TakeRows(dataView, 8);
preview = takeData.Preview();

//filter
var filterByValue = mlContext.Data.FilterRowsByColumn(dataView, nameof(InputModel.YearsOfExperience), lowerBound: 3, upperBound: 6);
preview = filterByValue.Preview();

//filter by missing value in salary, removed NaN value
var filterByMissingValue = mlContext.Data.FilterRowsByMissingValues(dataView, nameof(InputModel.Salary));
preview = filterByMissingValue.Preview();
```

### Data (Binary) Classification - Logistic regression example
* Supervised Learning - Fradulent detection, Medical Diagonsis, Spam Detection - 2 possible values -  True/False or 1/0 values
* BC Algorithms - Naive Bayes, Bayesian Classification, Decision Tree, Support Vector Machines, Neural Networks,..
* BC Popular Trainers - AveragedPerceptron, SdcaNonCalibrated, LbfgsLogisticRegression, SgdNonCalibrated, SdcaLogisticRegression, FastTree, LinearSvm, FastTree, Prior, Gam

|Manual|AutoML|
|-|-|
|test|tes1|

```csharp
var mlContext = new MLContext();

var dataView = mlContext.Data.LoadFromTextFile<TrainDelayInputModel>("flight_delay_train.csv", hasHeader: true, separatorChar: ',');

//used only needed columns in SelectColumns function - not included ORIGINAL_ARRIVAL_TIME,DELAY_MINUTES
var pipeline = mlContext.Transforms.SelectColumns(nameof(TrainDelayInputModel.Origin), nameof(TrainDelayInputModel.Destination),
                    nameof(TrainDelayInputModel.DepartureTime), nameof(TrainDelayInputModel.ExpectedArrivalTime),
                    nameof(TrainDelayInputModel.IsDelayBy15Minutes))

    //convert string to index, because ML.NET can't work with string
    //OneHotEncoding - Converts the Origin column(e.g., "NYC", "LAX") value into a binary vector.
    //Encoded_ORIGIN - The resulting encoded columns are stored under this new column name.
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Encoded_ORIGIN", nameof(TrainDelayInputModel.Origin)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Encoded_DESTINATION", nameof(TrainDelayInputModel.Destination)))

    //now we have 7 columns, so we can delete 2 original columns    
                .Append(mlContext.Transforms.DropColumns(nameof(TrainDelayInputModel.Origin), nameof(TrainDelayInputModel.Destination)))

                .Append(mlContext.Transforms.Concatenate("Features", "Encoded_ORIGIN", "Encoded_DESTINATION", nameof(TrainDelayInputModel.DepartureTime),
                                                            nameof(TrainDelayInputModel.ExpectedArrivalTime)))

                .Append(mlContext.Transforms.Conversion.ConvertType("Label", nameof(TrainDelayInputModel.IsDelayBy15Minutes), DataKind.Boolean))

                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

var model = pipeline.Fit(dataView);
var preview = model.Transform(dataView).Preview();

var predictionEngine = mlContext.Model.CreatePredictionEngine<TrainDelayInputModel, TrainDelayResultModel>(model);

var input = new TrainDelayInputModel { Origin = "JFK", Destination = "ATL", DepartureTime = 1930, ExpectedArrivalTime = 2225 };
var prediction = predictionEngine.Predict(input);

Console.WriteLine($"Prediction: {prediction.WillDelayBy15Minutes} | Score: {prediction.Score}");

input = new TrainDelayInputModel { Origin = "MSP", Destination = "SEA", DepartureTime = 1745, ExpectedArrivalTime = 1930 };
prediction = predictionEngine.Predict(input);

Console.WriteLine($"Prediction: {prediction.WillDelayBy15Minutes} | Score: {prediction.Score}");

  class TrainDelayResultModel
  {
      [ColumnName("PredictedLabel")]
      public bool WillDelayBy15Minutes { get; set; }

      public float Score { get; set; }
  }
  class TrainDelayInputModel
  {
      [LoadColumn(0)]
      public string Origin { get; set; }

      [LoadColumn(1)]
      public string Destination { get; set; }

      [LoadColumn(2)]
      public float DepartureTime { get; set; }

      [LoadColumn(3)]
      public float ExpectedArrivalTime { get; set; }

      [LoadColumn(4)]
      public float OriginalArrivalTime { get; set; }

      [LoadColumn(5)]
      public int DelayMinutes { get; set; }

      [LoadColumn(6)]
      public bool IsDelayBy15Minutes { get; set; }
  }
```
|
 ```csharp  
  var mlContext = new MLContext();
  var data =      return mlContext.Data.LoadFromTextFile<ModelInput>("flight_delay_train.csv", ',', hasHeader:true, allowQuoting: false);

  var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new []{new InputOutputColumnPair(@"ORIGIN", @"ORIGIN"),new InputOutputColumnPair(@"DESTINATION", @"DESTINATION")}, outputKind: OneHotEncodingEstimator.OutputKind.Indicator)      
                          .Append(mlContext.Transforms.ReplaceMissingValues(new []{new InputOutputColumnPair(@"DEPARTURE_TIME", @"DEPARTURE_TIME"),new InputOutputColumnPair(@"EXPECTED_ARRIVAL_TIME", @"EXPECTED_ARRIVAL_TIME")}))      
                          .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"ORIGIN",@"DESTINATION",@"DEPARTURE_TIME",@"EXPECTED_ARRIVAL_TIME"}))      
                          .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:@"IS_DELAY_BY_15_MINUTES",inputColumnName:@"IS_DELAY_BY_15_MINUTES",addKeyValueAnnotationsAsText:false))      
                          .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator:mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options(){NumberOfLeaves=4,MinimumExampleCountPerLeaf=20,NumberOfTrees=4,MaximumBinCountPerFeature=254,FeatureFraction=1,LearningRate=0.1,LabelColumnName=@"IS_DELAY_BY_15_MINUTES",FeatureColumnName=@"Features",DiskTranspose=false}),labelColumnName: @"IS_DELAY_BY_15_MINUTES"))      
                          .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:@"PredictedLabel",inputColumnName:@"PredictedLabel"));

  var model = pipeline.Fit(trainData);
```    
|


![image](https://github.com/user-attachments/assets/f1f9cb2e-7f34-46d1-809c-15bffb364eeb)

