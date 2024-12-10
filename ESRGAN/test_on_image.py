from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 设置固定的图像路径和模型路径
image_path = "./data/resized_val/airplane88.tif"  # 输入图片路径
checkpoint_model = "./saved_models/generator_best.pth"  # 模型路径

# 创建输出文件夹
os.makedirs("images/outputs", exist_ok=True)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义生成器模型并加载训练好的模型
generator = GeneratorRRDB(channels=3, filters=64, num_res_blocks=23).to(device)
generator.load_state_dict(torch.load(checkpoint_model, map_location=device))
generator.eval()

# 定义图像预处理和后处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# 加载并准备输入图像
image = Image.open(image_path).convert('RGB')
image = image.resize((128, 128))  # 确保输入图像为 128x128
image_tensor = Variable(transform(image)).to(device).unsqueeze(0)

# 超分辨率处理
with torch.no_grad():
    sr_image = generator(image_tensor)  # 生成超分辨率图像
    sr_image = torch.nn.functional.interpolate(sr_image, size=(256, 256), mode='bilinear', align_corners=False)  # 调整输出大小为 256x256
    sr_image = denormalize(sr_image).cpu()  # 反归一化

# 保存输出图像
fn = os.path.basename(image_path)  # 获取文件名
save_image(sr_image, f"images/outputs/sr-{fn}")

print(f"Super-resolution image saved at: images/outputs/sr-{fn}")
