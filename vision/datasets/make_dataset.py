import os

path = "C:/Users/kim hyun jun/data/VOCdevkit/VOC2007/ImageSets/Main"

def data_idx(path, in_txt, out_txt):
    with open(os.path.join(path, in_txt)) as f:
        labels = f.readlines()

    label_list = []
    print(labels[0].split())
    for i in labels:
        #print(i)
        label, exists = i.split()
        if exists == "-1" or exists == "0":
            continue
        label_list.append(label)

    print(len(label_list))

    with open(os.path.join(path, out_txt), "w") as f:
        for i in label_list:
            f.write(i+"\n")

data_idx(path, "chair_trainval.txt", "chair_trainval_asb.txt")

