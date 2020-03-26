import torch
import torch.nn as nn
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.asb_vgg import asb_vgg

from .ssd import SSD
from .predictor import Predictor
from .config import vgg_ssd_config as config


# class is only chair
# 다른 모델이 들어와도 유연하게 만들 수 있는 함수
# batch_normal 있는 거 생각
# assemble vgg에 꽁무니 친구들도 붙인다 o
# ceil mode True로 함요
# 227 >> 1024 괜찮을 지 생각
def create_assemble_vgg_ssd(num_classes, is_test=False):
    #base_net = torch.load(model_path).features[:-1]
    #base_net[23].ceil_mode = True

    vgg_config = [23, 34, "M", 63, 57, "M", 124, 107, 106, "C", 272, 217, 222, "M", 287, 235, 227]

    base_net = ModuleList(asb_vgg(vgg_config))
    # source_layer_indexes = [
    #     (23, BatchNorm2d(512)),
    #     len(base_net),
    # ]
    #make vgg_config
    # i = 0
    # for layer in base_net:
    #     try:
    #         vgg_config.append(layer.out_channels)
    #     except AttributeError:
    #         if layer.__str__()[0] == "M":
    #             i += 1
    #             if i == 3:
    #                 vgg_config.append("C")
    #             else:
    #                 vgg_config.append("M")
    #print(vgg_config)

    #fc
    # base_net = ModuleList(base_net)
    # pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # conv6 = nn.Conv2d(vgg_config[-1], 1024, kernel_size=3,
    #                   padding=6, dilation=6)
    # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # base_net.extend([pool5, conv6,
    #            nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)])

    source_layer_indexes = [
        (33, BatchNorm2d(vgg_config[12])),
        len(base_net)
    ]

    conv4_3_out_channels = vgg_config[-5]

    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=conv4_3_out_channels, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=conv4_3_out_channels, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        # TODO: change to kernel_size=1, padding=0?
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def create_assemble_vgg_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor

#create_assemble_vgg_ssd(21, "C:/Users/kim hyun jun/PycharmProjects/SSD_for_git/models/model.pth")
