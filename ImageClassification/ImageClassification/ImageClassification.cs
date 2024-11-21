using Microsoft.ML;
using Microsoft.ML.Data;

namespace POC_ML
{
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
}


