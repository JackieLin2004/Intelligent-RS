from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab


# 调整图片大小
def resize_img(img, HW=(256, 256), resample=3):
    """
    调整图像大小。

    参数：
    img: 输入的图像数组。
    HW: 一个元组，表示期望的图像高度和宽度，默认值为(256, 256)。
    resample: 图像重采样方法的标识符，默认值为3，对应于Image.BICUBIC。

    返回值：
    返回调整大小后的图像数组。
    """
    # 将输入的图像数组转换为PIL图像对象，以便进行图像处理操作
    # 使用resize方法调整图像大小，其中HW[1]代表宽度，HW[0]代表高度
    # resample参数指定图像重采样方法，这里使用默认值3，对应于双三次插值法
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


class ColorData(Dataset):
    def __init__(self, img_paths, train=1):
        """
        初始化数据集类的构造函数。

        根据train参数的值，决定是使用训练集还是测试集。如果train参数为1，则使用训练集，并应用一系列数据增强变换；
        如果train参数为0，则仅使用测试集。此外，保存train参数的值，以便于后续使用。

        参数:
        - train: int, 指定使用训练集(1)还是测试集(0)的标志。
        """
        self.images = []

        for img_path in img_paths:
            img = Image.open(img_path)
            img = np.array(img)
            resized_img = resize_img(img)
            self.images.append(resized_img)

        # 根据train参数的值选择数据集和变换方法
        if train == 1:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])
        elif train == 0:
            self.transforms  = None

        # 保存train参数的值，以便于后续使用
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx]).convert('RGB')
        lab_imgs = rgb2lab(img).astype("float32")
        lab_imgs = transforms.ToTensor()(lab_imgs)
        L = lab_imgs[[0], ...] / 50. - 1.
        ab = lab_imgs[[1, 2], ...] / 110.

        return L, ab
