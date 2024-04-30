import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import polynomial_kernel


def calculate_kid(real_images, generated_images):
    # 加载Inception V3模型
    inception = InceptionV3(include_top=False, pooling="avg")
    # 预处理图像
    real_images = preprocess_input(real_images)
    generated_images = preprocess_input(generated_images)
    # 提取真实图像和生成图像的特征向量
    real_features = inception.predict(real_images)
    generated_features = inception.predict(generated_images)
    # 计算特征向量之间的核矩阵
    kernel_matrix = polynomial_kernel(real_features, generated_features)
    # 计算核矩阵的Fréchet距离
    kid = (
        np.trace(kernel_matrix)
        + np.trace(real_features)
        + np.trace(generated_features)
        - 3 * np.trace(sqrtm(kernel_matrix))
    )
    return kid


# 示例用法
real_images = np.random.rand(100, 299, 299, 3)  # 100张真实图像
generated_images = np.random.rand(100, 299, 299, 3)  # 100张生成图像
kid = calculate_kid(real_images, generated_images)
print("KID: ", kid)
