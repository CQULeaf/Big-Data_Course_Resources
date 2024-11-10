import base64
import io
import numpy as np
from PIL import Image
from PIL import ImageOps
from django.http import JsonResponse
from django.shortcuts import render
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
import cv2

# HOG 参数
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'
visualize = False
transform_sqrt = False
feature_vector = True
multichannel = None

# start spark local and load model
spark = SparkSession.builder \
    .appName("load_and_predict") \
    .master("local") \
    .getOrCreate()

# 加载训练时保存的模型
model = LogisticRegressionModel.load("/home/ecs-user/Big-Data_Course_Resources/Lab/Lab3_CHN/result")

# 汉字数字映射
code_to_hanzi = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']

def handwrittenBoard(request):
    # 判断是否是 AJAX 请求
    if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        # 获取图像数据并解码
        img_not_decode = request.GET.getlist('img')
        img_decode = base64.urlsafe_b64decode(img_not_decode[0])
        image = io.BytesIO(img_decode)
        img = Image.open(image)
        
        img = img.resize((64, 64), Image.Resampling.LANCZOS)

        # 处理图像
        img2 = img.convert("RGB")  # 确保转换为 RGB 模式，而不是 RGBA
        img = ImageOps.invert(img2)  # 图像反转
        img.save("./imgImageOpInvert.jpg")  # 保存为 JPEG 格式


        # 转为灰度图像并转换为 NumPy 数组
        my_image = img.convert('L')
        img_arr = np.array(my_image)

        # 使用 OpenCV 计算 HOG 特征
        hog_descriptor = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=orientations
        )

        # 计算 HOG 特征
        features = hog_descriptor.compute(img_arr)
        features = features.flatten()

        # 将特征转换为 Spark 的向量格式
        features_spark = Vectors.dense(features)

        # 将特征包装成 DataFrame
        features_df = spark.createDataFrame([(features_spark,)], ["features"])

        # 使用模型进行预测
        prediction = model.transform(features_df)
        prediction_result = prediction.collect()[0]["prediction"]
        
        # 获取对应的汉字
        res = code_to_hanzi[int(prediction_result)]

        # 返回结果
        response = JsonResponse({"res": res})
        return response

    return render(request, 'handwrittenBoard.html')
