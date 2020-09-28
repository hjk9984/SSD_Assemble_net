import os
import matplotlib.pyplot as plt

def visualize_result(output_path, data_path, output_name):
    data_list = os.listdir(data_path)
    epochs_list = []
    wgt_list = []
    epoch_idx = 0

    temp = data_list[0].split("-")
    for idx, i in enumerate(temp):
        if i == "Epoch":
            epoch_idx = idx + 1
            break

    for data in data_list:
        str_list = data.split("-")
        epochs_list.append(int(str_list[epoch_idx]))
        wgt_list.append(float(str_list[-1][:-4]))

    plt.figure(figsize=(15, 10))
    plt.grid()
    plt.scatter(epochs_list, wgt_list, label=output_name[:-7])
    plt.title(output_name)
    plt.ylim((3.0, 5.5))
    plt.savefig(os.path.join(output_path, output_name))
    plt.show()
    plt.exit()


visualize_result("./", "./models/vgg_bn", "vgg_bn_result")
