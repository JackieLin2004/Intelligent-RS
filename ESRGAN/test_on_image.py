from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 直接指定路径和参数
image_path = "data/airplane_val/airplane98.tif"  # 替换为你的图片路径
checkpoint_model = "saved_models/generator_best.pth"  # 替换为你的模型路径
channels = 3  # 通常RGB图像是3个通道
residual_blocks = 23  # 默认23个残差块

# 创建输出目录
os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型并加载检查点
generator = GeneratorRRDB(channels, filters=64, num_res_blocks=residual_blocks).to(device)
generator.load_state_dict(torch.load(checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# 准备输入
image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)

# 上采样图像
with torch.no_grad():
    sr_image = denormalize(generator(image_tensor)).cpu()

# 保存图像
fn = image_path.split("/")[-1]
save_image(sr_image, f"images/outputs/sr-{fn}")

print(f"Super-resolved image saved as: images/outputs/sr-{fn}")
