import os
#import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
#from dataset.data_loader import GetLoader
from torchvision import datasets
import numpy as np


def test(dataset_name, dataloader, epoch):
    assert dataset_name in ['src_7_0','tar_7_1', 'tar_7_2', 'tar_7_3', 'tar_14_1', 'tar_14_2', 'tar_14_3', 
                            'tar_21_1', 'tar_21_2', 'tar_21_3', 'FE_tar_7_1', 'FE_tar_7_2', 'FE_tar_7_3', 
                            'FE_tar_14_1', 'FE_tar_14_2', 'FE_tar_14_3', 
                            'FE_tar_21_1', 'FE_tar_21_2', 'FE_tar_21_3', 'ims_tar']

    model_root = os.path.join('.', 'models')
#    image_root = os.path.join('.', 'dataset', dataset_name)

#    cuda = False
#    cudnn.benchmark = False
    batch_size = 5
    image_size = 10
    alpha = 0

    """load data"""
    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

#    if cuda:
#        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    h_features = torch.tensor(np.array([]), dtype = torch.float32)
    predLabels = torch.tensor(np.array([]), dtype = torch.long)

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

#        if cuda:
#            t_img = t_img.cuda()
#            t_label = t_label.cuda()
#            input_img = input_img.cuda()
#            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, h_feature , _ = my_net(input_data = input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        h_features = torch.cat((h_features, h_feature),0)
        predLabels = torch.cat((predLabels, pred), 0)

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu, h_features, predLabels
