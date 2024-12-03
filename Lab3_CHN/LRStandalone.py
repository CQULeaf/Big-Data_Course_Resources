from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import tempfile

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("ChineseHandwritingNumber") \
    .master("spark://master:7077") \
    .getOrCreate()

# 设置日志级别为 WARN 以减少日志输出
spark.sparkContext.setLogLevel("WARN")
print("Load Spark successful")

# 定义 HDFS 中的训练和测试数据路径
TRAINPATH = "/chn/train.csv"
TESTPATH = "/chn/test.csv"

# 读取训练数据和测试数据
# 假设 CSV 文件没有标题行，如果有标题行，请添加 header=True
train_df = spark.read.csv(TRAINPATH, header=False, inferSchema=True)
test_df = spark.read.csv(TESTPATH, header=False, inferSchema=True)

print("Load HDFS data successful")

# 假设最后一列是标签，前面的列是特征
# 获取特征列的名称（例如 _c0, _c1, ..., _cN-1）
feature_columns = train_df.columns[:-1]
label_column = train_df.columns[-1]

# 使用 VectorAssembler 将特征列合并为一个特征向量
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_assembled = assembler.transform(train_df).select(col(label_column).alias("label"), "features")
test_assembled = assembler.transform(test_df).select(col(label_column).alias("label"), "features")

print("Data transformation successful")

# 初始化逻辑回归模型
# 设置最大迭代次数和类别数
lr = LogisticRegression(maxIter=100, regParam=0.0, elasticNetParam=0.0, featuresCol="features", labelCol="label", family="multinomial")

# 训练模型
print("Model training started at:", time.strftime('%Y-%m-%d %H:%M:%S'))
model = lr.fit(train_assembled)
print("Model training completed at:", time.strftime('%Y-%m-%d %H:%M:%S'))

# 保存模型到临时目录
# 你可以更改路径以保存到 HDFS 或其他持久存储
path = tempfile.mkdtemp()
model.save(path)
print("Model saved at:", path)

# 对测试数据进行预测
predictions = model.transform(test_assembled)

# 评估模型准确率
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# 停止 SparkSession
spark.stop()
