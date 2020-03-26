from ..transforms.transforms import *

def get_items(img=None, boxes=None, labels=None):
    return (img/1.0, boxes, labels)

class ImgNormalization(object):
    def __init__(self, std):
        self.std = std

    def __call__(self, img, boxes=None, labels=None):
        #img, boxes, labels = sample["img"], sample["boxes"], sample["labels"]
        return img/self.std, boxes, labels


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            #get_items(),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ImgNormalization(std),
            ToTensor(),
        ])


    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            #get_items(),
            ImgNormalization(std),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            #get_items(),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ImgNormalization(std),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image