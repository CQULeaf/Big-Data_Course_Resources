from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
import tempfile

# 配置 SparkConf 使用 YARN 模式
conf = SparkConf() \
    .setAppName("CHN") \
    .setMaster("yarn") \
    .set("spark.submit.deployMode", "cluster") \
    .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./big_data_lab_env/bin/python") \
    .set("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", "./big_data_lab_env/bin/python") \
    .set("spark.archives", "hdfs:///shared/envs/big_data_lab_env.tar.gz#big_data_lab_env")

# 初始化 SparkSession
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark 配置成功")

TRAIN_PATH = "/chn/train.csv"
TEST_PATH = "/chn/test.csv"

print("-----加载训练数据-----")
# 加载训练数据并转换为 DataFrame 格式
train_rdd = spark.sparkContext.textFile(TRAIN_PATH)
train_data = train_rdd.map(lambda line: Row(
    label=float(line.split(",")[-1]),
    features=[float(x) for x in line.split(",")[:-1]]
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
    features=[float(x) for x in line.split(",")[:-1]]
)).toDF()

# 将特征转换为单个向量列
test_data = vector_assembler.transform(test_data).select("features_vec", "label")
print("-----测试数据加载成功-----")

# 使用 ml 的 LogisticRegression
lr = LogisticRegression(featuresCol="features_vec", labelCol="label", maxIter=100, regParam=0.3, elasticNetParam=0.8)
model = lr.fit(train_data)

print("-----模型训练成功-----")

# 保存模型
model_save_path = tempfile.mkdtemp()
model.save(model_save_path)
print("-----模型保存成功-----")

print("-----模型评估中-----")
# 使用测试数据进行预测并计算准确率
predictions = model.transform(test_data)
correct = predictions.filter(predictions.label == predictions.prediction).count()
total = test_data.count()
accuracy = correct / total
print("模型准确率:", accuracy)

spark.stop()
