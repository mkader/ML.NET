### Data (Binary) Classification - Sentiment Analysis - Imdb Movie Review Example

* Manual
```csharp
var mlContext = new MLContext();

var dataset = mlContext.Data.LoadFromTextFile<InputModel>(path: @"imdb_reviews_train.tsv", hasHeader: true );

//FeaturizeText() - converts raw text data into numerical features for use in ML models.
//sample dataset: "I love programming", "ML.NET is awesome", "Text analysis is fun"
//The transformation FeaturizeText converts SentimentText into a numerical vector like: Features(Vector)
//[0.1, 0.3, 0.5, 0.7, ...],  [0.2, 0.4, 0.6, 0.8, ...], [0.3, 0.5, 0.7, 0.9, ...]
//These vectors can then be fed into a ML model for tasks like sentiment analysis or text classification.
var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(InputModel.SentimentText))

//The Averaged Perceptron algorithm, a linear classification method suitable for binary classification tasks(e.g., positive vs.negative).
//The number of passes (iterations) the trainer will make over the dataset to optimize the model.
                .Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: nameof(InputModel.Sentiment), numberOfIterations: 100));

var model = pipeline.Fit(dataset);

var predictionEngine = mlContext.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

PrintResult(predictionEngine, "I liked this movie.");
PrintResult(predictionEngine, "Movie was just ok.");
PrintResult(predictionEngine, "It's a really good film. Outragously entertaining, loved it.");
PrintResult(predictionEngine, "It's worst movie I have ever seen. Boring, badly written.");


static void PrintResult(PredictionEngine<InputModel, ResultModel> predictionEngine, string sentimentText)
{
    var input = new InputModel { SentimentText = sentimentText };

    var result = predictionEngine.Predict(input);
    
    Console.WriteLine($"Prediction: {result.IsPositiveReview} | Score: {result.Score}");
}
 
class ResultModel
{
    [ColumnName("PredictedLabel")]
    public bool IsPositiveReview { get; set; }

    public float Score { get; set; }
}

class InputModel
{
    [LoadColumn(0)]
    public bool Sentiment { get; set; }

    [LoadColumn(1)]
    public string SentimentText { get; set; }
}
```
* AutoML - BinaryClassification, all datas are trained
```csharp
// Data process configuration with pipeline data transformations
var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName:@"SentimentText",outputColumnName:@"SentimentText")      
                        .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"SentimentText"}))      
                        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:@"Sentiment",inputColumnName:@"Sentiment",addKeyValueAnnotationsAsText:false))      
                        .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator:mlContext.BinaryClassification.Trainers.FastForest(new FastForestBinaryTrainer.Options(){NumberOfTrees=4,NumberOfLeaves=4,FeatureFraction=1F,LabelColumnName=@"Sentiment",FeatureColumnName=@"Features"}),labelColumnName:@"Sentiment"))      
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:@"PredictedLabel",inputColumnName:@"PredictedLabel"));
```

* AutoML - TextClassification, only trained first 20 rows, it will take more if all datasets to train
```csharp
 // Data process configuration with pipeline data transformations
 var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:@"Sentiment",inputColumnName:@"Sentiment",addKeyValueAnnotationsAsText:false)      
                         .Append(mlContext.MulticlassClassification.Trainers.TextClassification(labelColumnName: @"Sentiment", sentence1ColumnName: @"SentimentText"))      
                         .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:@"PredictedLabel",inputColumnName:@"PredictedLabel"));

public class ModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"Sentiment")]
    public float Sentiment { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"SentimentText")]
    public string SentimentText { get; set; }

}

public class ModelOutput
{
    [ColumnName(@"Sentiment")]
    public uint Sentiment { get; set; }

    [ColumnName(@"SentimentText")]
    public float[] SentimentText { get; set; }

    [ColumnName(@"Features")]
    public float[] Features { get; set; }

    [ColumnName(@"PredictedLabel")]
    public float PredictedLabel { get; set; }

    [ColumnName(@"Score")]
    public float[] Score { get; set; }
}
```
