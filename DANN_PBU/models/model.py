import torch.nn as nn
from models.functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self, imageSize = 10):
        super(CNNModel, self).__init__()
        self.imageSize = imageSize
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 32, kernel_size=3))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(32, 100, kernel_size=7))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(100))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.fc_layers = nn.Sequential()
        self.fc_layers.add_module('c_fc1', nn.Linear(100, 50))
        self.fc_layers.add_module('c_bn1', nn.BatchNorm1d(50))
        self.fc_layers.add_module('c_relu1', nn.ReLU(True))
        self.fc_layers.add_module('c_drop1', nn.Dropout2d())
        self.fc_layers.add_module('c_fc2', nn.Linear(50, 20))
        self.fc_layers.add_module('c_bn2', nn.BatchNorm1d(20))
        self.fc_layers.add_module('c_relu2', nn.ReLU(True))
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc3', nn.Linear(20, 3))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(100, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 3))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 1, self.imageSize, self.imageSize)
        feature = self.feature(input_data)
        print(feature.shape)
        feature = feature.view(-1, 100)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        h_feature = self.fc_layers(feature)
#        print(h_feature.shape)
        class_output = self.class_classifier(h_feature)
        domain_output = self.domain_classifier(reverse_feature)
#        h_feature = np.reshape(h_feature, )

        return class_output, h_feature, domain_output
    

#my_net = CNNModel()

#class_output, h_feature , domain_output = my_net(input_data = input_img, alpha=alpha)
