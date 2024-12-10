import os
from PIL import Image

# 设置源文件夹和目标文件夹
source_folder = '../../data/airplane'  # 原始图片文件夹
output_folder = '../../data/resized'  # 新的保存文件夹

# 如果目标文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历源文件夹中的所有图片文件
for filename in os.listdir(source_folder):
    # 构造图片的完整路径
    file_path = os.path.join(source_folder, filename)

    # 判断是否是图片文件
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
        # 打开图片
        with Image.open(file_path) as img:
            # 确保图片尺寸是256x256
            if img.size == (256, 256):
                # 调整图片大小到128x128
                resized_img = img.resize((128, 128))
                # 构造保存路径
                save_path = os.path.join(output_folder, filename)
                # 保存调整后的图片
                resized_img.save(save_path)
                print(f"Image {filename} resized and saved to {save_path}")
            else:
                print(f"Skipping {filename}, as it is not 256x256.")
