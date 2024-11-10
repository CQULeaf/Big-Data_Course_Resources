from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# 配置 SparkConf 使用 yarn 模式
conf = SparkConf() \
    .setAppName("CHN_STANDALONE")
    # .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./big_data_lab_env/bin/python") \
    # .set("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", "./big_data_lab_env/bin/python") \
    # .set("spark.archives", "hdfs:///shared/envs/big_data_lab_env.tar.gz#big_data_lab_env")

# 初始化 SparkSession
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark Standalone 模式配置成功")

TRAIN_PATH = "/chn/train.csv"
TEST_PATH = "/chn/test.csv"
MODEL_SAVE_DIR = "hdfs:///chn/model"

print("-----加载训练数据-----")
# 加载训练数据并转换为 DataFrame 格式
train_rdd = spark.sparkContext.textFile(TRAIN_PATH)
train_data = train_rdd.map(lambda line: Row(
    label=float(line.split(",")[-1]),
    features=Vectors.dense([float(x) for x in line.split(",")[:-1]])
)).toDF()

# 将特征转换为单个向量列
vector_assembler = VectorAssembler(inputCols=["features"], outputCol="features_vec")
train_data = vector_assembler.transform(train_data).select("features_vec", "label")

print("-----训练数据加载成功-----")

print("-----加载测试数据-----")
# 加载测试数据并转换为 DataFrame 格式
test_rdd = spark.sparkContext.textFile(TEST_PATH)
test_data = test_rdd.map(lambda line: Row(
    label=float(line.split(",")[-1]),
    features=Vectors.dense([float(x) for x in line.split(",")[:-1]])
)).toDF()

# 将特征转换为单个向量列
test_data = vector_assembler.transform(test_data).select("features_vec", "label")
print("-----测试数据加载成功-----")

# 定义评估器，使用多分类的准确率评估
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# 定义多个模型
models = {
    "RandomForest": RandomForestClassifier(featuresCol="features_vec", labelCol="label", numTrees=50),
    "GradientBoostedTrees": GBTClassifier(featuresCol="features_vec", labelCol="label", maxIter=100),
    "MultilayerPerceptron": MultilayerPerceptronClassifier(featuresCol="features_vec", labelCol="label", maxIter=100, layers=[train_data.schema["features_vec"].metadata["ml_attr"]["num_attrs"], 128, 64, 15])
}

# 存储最高准确率及其对应模型
best_accuracy = 0.0
best_model = None
best_model_name = ""

print("-----开始训练和评估模型-----")

for model_name, model in models.items():
    print(f"训练模型：{model_name}")
    trained_model = model.fit(train_data)
    print(f"{model_name} 模型训练完成")

    # 在测试数据上评估模型
    predictions = trained_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    print(f"{model_name} 模型准确率: {accuracy:.4f}")

    # 如果当前模型准确率更高，则更新最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = trained_model
        best_model_name = model_name

print("-----所有模型训练和评估完成-----")
print(f"最佳模型为：{best_model_name}，准确率为：{best_accuracy:.4f}")

# 保存最佳模型
best_model_save_path = os.path.join(MODEL_SAVE_DIR, best_model_name)
best_model.save(best_model_save_path)
print(f"最佳模型已保存至：{best_model_save_path}")

spark.stop()
