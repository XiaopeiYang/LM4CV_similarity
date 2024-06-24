import numpy as np
import matplotlib.pyplot as plt
import os

# 提供的数据
data = np.array([
    0.25787535, 0.25376663, 0.26030096, 0.2506538, 0.24973561, 0.2201587,
    0.21540648, 0.2694801, 0.2506395, 0.2519347, 0.27566513, 0.2597009,
    0.23528072, 0.25250983, 0.23057388, 0.24474818, 0.24731052, 0.25954992,
    0.24925315, 0.2534912, 0.24760443, 0.22211061, 0.24586664, 0.23736443,
    0.24949312, 0.26298806, 0.26749083, 0.26597893, 0.257297, 0.23509617,
    0.253054, 0.2458682, 0.23956563, 0.24606594, 0.25340882, 0.26047748,
    0.24689199, 0.24875312, 0.20436785, 0.25865188, 0.27448833, 0.23674037,
    0.24137492, 0.22865793, 0.2767707, 0.21413434, 0.24973357, 0.271344,
    0.22984278, 0.26529962, 0.22602376, 0.26296726, 0.21769398, 0.2654812,
    0.25253263, 0.25283954, 0.2300297, 0.2544265, 0.2674244, 0.253171,
    0.25588945, 0.22988422, 0.22568911, 0.26623327
])

# 立方变换
cubed_data = np.power(data, 3)

# 自定义非线性变换: 例如将对数和指数变换结合起来
custom_transformed_data = np.exp(np.log(data) * 3)
print(custom_transformed_data)

# 创建保存图片的文件夹
output_dir = "/export/home/ra79nod/proj/LM4CV_similarity/transform_plot"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 绘制原始数据与变换后数据的对比并保存图片
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.bar(range(len(data)), data)
plt.title('Original Data')
plt.xlabel('Index of Attributes')
plt.ylabel('Scores')

plt.subplot(2, 2, 2)
plt.bar(range(len(cubed_data)), cubed_data, color='orange')
plt.title('Transformed Data (Cubed)')
plt.xlabel('Index of Attributes')
plt.ylabel('Transformed Scores')

plt.subplot(2, 2, 3)
plt.bar(range(len(custom_transformed_data)), custom_transformed_data, color='green')
plt.title('Transformed Data (Custom Non-linear)')
plt.xlabel('Index of Attributes')
plt.ylabel('Transformed Scores')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "transformed_plots_v2.png"))
plt.show()
