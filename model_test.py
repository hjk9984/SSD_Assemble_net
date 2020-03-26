import torch
import torchvision.models as mm

base = torch.load("models/model.pth")
torch.save(base.features[:43].state_dict(), "./models/pretrained_asb_vgg.pth")

config = [23, 34, "M", 63, 57, "M", 124, 107, 106, "M", 272, 217, 222, "M", 287, 235, 227]

# torch.save(mm.vgg16(pretrained=True).features[:24].state_dict(), "./models/pretrained_vgg16.pth")
#
#
# class test_module(torch.nn.Module):
#     def __init__(self, input_ch, output_ch):
#         super(test_module, self).__init__()
#         self.conv23 = torch.nn.Conv2d(input_ch,output_ch,3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#
#
# net = test_module(3, 32)
# print(net)
# net1 = test_module(32, 64)
# print(net1)
# print("modules----------------")
# net.add_module("conv2", net1)
# for i in net.modules():
#     print(i)
#
# print("children-----------------")
# for i in net.children():
#     print(i)
#
# print("named_children-=----")
# for name, i in net.named_children():
#     print(name,"////", i)
#
# for key in net.state_dict(prefix="features.", keep_vars=True):
#     print(key)
#
# for i in net.parameters(recurse=False):
#     print(i)


"""
temp = torch.load("./models/model.pth")
print(temp)
print(type(temp.features))
temp1 = torch.nn.Conv2d(32,32,3,2)
print(temp1.__str__())

print(temp.features[33])
temp.features[33].ceil_mode=True
print(temp.features[33])

print(type(temp.features[0].out_channels))

temp = mm.vgg16(pretrained=True)
torch.save(temp.state_dict(), "./models/vgg16_pretrained.pth")
"""


#temp

# k = 0
# for i in temp.features:
#     k += 1
#     try:
#         print(i.out_channels)
#     except AttributeError:
#         if i.__str__()[0] =="B":
#             print("B")
#         elif i.__str__()[0] =="R":
#             print("R")
#         elif i.__str__()[0] =="M":
#             print("M")
#         #print(i)
#
# print(k)