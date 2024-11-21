## Imaging dataset
* Use the image dataset (ImageClassificationDataset.zip), contains the images that shall be used for training within the sample-images folder.
* The dataset consists of images from another <a href="https://github.com/dotnet/samples/blob/main/machine-learning/tutorials/TransferLearningTF/image-classifier-assets.zip">GitHub repository</> and pictures from Pixabay
* The images to test the model are located within the test-images folder.
* The training images have been placed into subfolders (found within the sample-images folder), each representing an image classification category. 
  1. ![image](https://github.com/user-attachments/assets/6655bdb5-6836-4146-9750-2f2188467168)
  1. urls.txt file within the sample-images folder. This file indicates the sources from where each image was obtained (this GitHub repository and Pixabay).
* When working with image classification in ML.NET, the images used to train the model must be placed within a specific subfolder structure, where each folder indicates the image category associated with a particular image type.
  1. Like, the Pizza subfolder contains pizza images, the Broccoli subfolder contains pictures of broccoli, and so on—these subfolders represent image labels.

## Generating the model - ImageClassificationModel.mbconfig
*  Console application -> select Image Classification scenario -> Add Data -> Select training images folder ("sample-images")
    1. the images and their different categories will be shown within the Data Preview section
    1. ![image](https://github.com/user-attachments/assets/3b85e41c-0bea-43e3-bf15-1e87aa851fe0)
*  Start training -> Evaluate -> evaluate all images from "test-image" folder.
    1. Model Builder correctly predicts that the image corresponds(example girl-447701_640.jpg -> Teddy Bear category).
    1. The girl image shows that the Teddy Bear category has been given a score of 88% compared to the other types. 2 things to pay close attention.
    2. 1st, only trained each image category with five sample images.
    3. 2nd, the sample Teddy Bear training images only have a teddy bear and nothing else. However, the test image contains a road, a little girl, and a teddy bear.
    1. ![image](https://github.com/user-attachments/assets/c2990a03-1b1c-4361-b19f-e73fc0c2e378)
    2. ![image](https://github.com/user-attachments/assets/e727ced5-97a6-4188-a3e1-cc875fcf4730)
    3. ![image](https://github.com/user-attachments/assets/d659389a-15aa-42a8-aca1-f9d08663d665)
    4. ![image](https://github.com/user-attachments/assets/669f3c2d-7e1c-41bb-bc89-171e7cc17b12)
* Copy the code snippet, Run the code, couple of errors are associated with it.
```csharp
Console.WriteLine("Hello, World!");

//Load sample data
var imageBytes = File.ReadAllBytes(@"C:\ImageClassificationDataset\sample-images\Broccoli\bowl-of-broccoli-2584307_640.jpg");
ImageClassificationModel.ModelInput sampleData = new ImageClassificationModel.ModelInput()
{
    ImageSource = imageBytes,
};

//Load model and predict output
var icResult = ImageClassificationModel.Predict(sampleData);
Console.WriteLine($"Category {icResult.PredictedLabel}");
```
  1. ![image](https://github.com/user-attachments/assets/d80fb65a-e16d-4217-8f70-76a96e886396)

## Write code manually
* mlContext.MulticlassClassification.Trainers.ImageClassification cannot be found, because missing Microsoft.ML.Vision NuGet package
* Fixing the missing TensorFlow DLL. TensorFlow is an open-source and freely available ML platform from Google. Internally, ML.NET uses TensorFlow to perform image classification.
* To install TensorFlow for ML.NET, SciSharp.TensorFlow.Redist NuGet Package
```csharp
ImageClassification.cs
using Microsoft.ML;
using Microsoft.ML.Data;

  public class ImageClassification
  {
      public class ModelInput
      {
          [LoadColumn(0)]
          [ColumnName(@"Label")]
          public string Label { get; set; }
          [LoadColumn(1)]
          [ColumnName(@"ImageSource")]
          public byte[] ImageSource { get; set; }
      }

      public class ModelOutput
      {
          [ColumnName(@"Label")]
          public uint Label { get; set; }
          [ColumnName(@"ImageSource")]
          public byte[] ImageSource { get; set; }
          [ColumnName(@"PredictedLabel")]
          public string PredictedLabel { get; set; }
          [ColumnName(@"Score")]
          public float[] Score { get; set; }
      }

      private static string MLNetModelPath = Path.GetFullPath("ImageClassificationModel.mlnet");

      public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy <PredictionEngine<ModelInput, ModelOutput>> (() => CreatePredictEngine(), true);

      private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
      {
          var mlContext = new MLContext();
          ITransformer mlModel =  mlContext.Model.Load(MLNetModelPath, out var _);
          return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
      }

      public static ModelOutput Predict(ModelInput input)
      {
          var predEngine = PredictEngine.Value;
          return predEngine.Predict(input);
      }

      public static IEstimator<ITransformer> BuildPipeline(
        MLContext mlContext)
      {
          var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label")
          .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label", scoreColumnName: @"Score", featureColumnName: @"ImageSource"))
          .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));
          return pipeline;
      }

      public static ITransformer RetrainPipeline(MLContext mlContext, IDataView trainData)
      {
          var pipeline = BuildPipeline(mlContext);
          var model = pipeline.Fit(trainData);
          return model;
      }
}

Program.cs
var icSampleData = new ImageClassification.ModelInput()
{
    ImageSource = imageBytes,
};

var icResult = ImageClassification.Predict(icSampleData);
Console.WriteLine($"IC Category {icResult.PredictedLabel}");

```    
* ModelInput class
  1. The Label property represents the category to which the training image belongs, in other words, the subfolder where the image resides.
  2. The ImageSource property indicates the file path to the training image.
* ModelOutput
  1. The PredictedLabel property will indicate what category a nontraining image will be given once processed by the ML algorithm that the model will utilize.
  2. The ML algorithm will use the Score property that the model will utilize to indicate how confident the ML algorithm is of the category (PredictedLabel) assigned to the image.
* MLNetModelPath - The file path to the future model’s metadata file
* Prediect method responsible for predicting the model’s results
* Build the pipeline
  1. MapValueToKey - Convert categorical (string) values into numerical ones. Input values are numeric (pixels), but in this example, the labels, such as broccoli, are strings that must be converted to integers.
  3. ImageClassification - Specify the training algorithm. Pass the Label column, the column containing the confidence percentage of the result obtained (Score), and the column specifying the features of the image (which is the image itself, ImageSource).
  4. MapKeyToValue- Convert the key types back to their original (categorical) values.   
