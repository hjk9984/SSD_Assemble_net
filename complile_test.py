from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.data_preprocessing import TestTransform, TrainAugmentation
from vision.ssd.config import vgg_ssd_config
import matplotlib.pyplot as plt
config = vgg_ssd_config
dataset = VOCDataset("C:/Users/kim hyun jun/data/VOCdevkit/VOC2007", is_test=False, keep_difficult=False,
                    transform=TrainAugmentation(config.image_size, config.image_mean, config.image_std))
data, box, label = next(iter(dataset))
plt.imshow(data.reshape(300, 300, 3))

print(data[0][0])
print(data.dtype)
print(box[0])
print(box.dtype)
print(label.dtype)