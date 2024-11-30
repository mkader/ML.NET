### Data (Binary) Classification - Logistic regression example
* Supervised Learning - Fradulent detection, Medical Diagonsis, Spam Detection - 2 possible values -  True/False or 1/0 values
* BC Algorithms - Naive Bayes, Bayesian Classification, Decision Tree, Support Vector Machines, Neural Networks,..
* BC Popular Trainers - AveragedPerceptron, SdcaNonCalibrated, LbfgsLogisticRegression, SgdNonCalibrated, SdcaLogisticRegression, FastTree, LinearSvm, FastTree, Prior, Gam

* Manual Code
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
* AuotML Code
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

 public class ModelInput
 {
     [LoadColumn(0)]
     [ColumnName(@"ORIGIN")]
     public string ORIGIN { get; set; }

     [LoadColumn(1)]
     [ColumnName(@"DESTINATION")]
     public string DESTINATION { get; set; }

     [LoadColumn(2)]
     [ColumnName(@"DEPARTURE_TIME")]
     public float DEPARTURE_TIME { get; set; }

     [LoadColumn(3)]
     [ColumnName(@"EXPECTED_ARRIVAL_TIME")]
     public float EXPECTED_ARRIVAL_TIME { get; set; }

     [LoadColumn(6)]
     [ColumnName(@"IS_DELAY_BY_15_MINUTES")]
     public float IS_DELAY_BY_15_MINUTES { get; set; }

 }


 public class ModelOutput
 {
     [ColumnName(@"ORIGIN")]
     public float[] ORIGIN { get; set; }

     [ColumnName(@"DESTINATION")]
     public float[] DESTINATION { get; set; }

     [ColumnName(@"DEPARTURE_TIME")]
     public float DEPARTURE_TIME { get; set; }

     [ColumnName(@"EXPECTED_ARRIVAL_TIME")]
     public float EXPECTED_ARRIVAL_TIME { get; set; }

     [ColumnName(@"IS_DELAY_BY_15_MINUTES")]
     public uint IS_DELAY_BY_15_MINUTES { get; set; }

     [ColumnName(@"Features")]
     public float[] Features { get; set; }

     [ColumnName(@"PredictedLabel")]
     public float PredictedLabel { get; set; }

     [ColumnName(@"Score")]
     public float[] Score { get; set; }

 }
```    
![image](https://github.com/user-attachments/assets/f1f9cb2e-7f34-46d1-809c-15bffb364eeb)
