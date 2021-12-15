import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import pt_util


def img_transpose(img):
    x = torch.transpose(img, 0, 1)
    x = torch.transpose(x, 1, 2)
    return x

def to_RGB(img):
    return np.array(img.convert('RGB'))

def rgb_to_label(mask):
    result = np.zeros((mask.shape[0], mask.shape[1]))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color = VOC_COLORMAP.get((mask[i, j, 0], mask[i, j, 1], mask[i, j, 2]), 0)
            result[i, j] = color
    return result

# given the label, this method will return a one hot encoding of it
def rgb_to_onehot(mask):
    result = np.zeros((mask.shape[0], len(VOC_CLASSES), mask.shape[1], mask.shape[2]))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                result[i, mask[i, j, k], j, k] = 1
    return result

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORMAP = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (0, 128, 0): 2,
    (128, 128, 0): 3,
    (0, 0, 128): 4,
    (128, 0, 128): 5,
    (0, 128, 128): 6,
    (128, 128, 128): 7,
    (64, 0, 0): 8,
    (192, 0, 0): 9,
    (64, 128, 0): 10,
    (192, 128, 0): 11,
    (64, 0, 128): 12,
    (192, 0, 128): 13,
    (64, 128, 128): 14,
    (192, 128, 128): 15,
    (0, 64, 0): 16,
    (128, 64, 0): 17,
    (0, 192, 0): 18,
    (128, 192, 0): 19,
    (0, 64, 128): 20,
}

class VGG16SegmentNet(nn.Module):
    def __init__(self):
        super(VGG16SegmentNet, self).__init__()
        self.test_loss = None
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=True)
        self.fc6 = nn.Linear(7*7*512, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.deconv_fc6 = nn.Linear(4096, 7*7*512) # and then we reshape this to be 7x7x512
        self.unpool5 = nn.MaxUnpool2d(2, stride=2, padding=0)
        self.deconv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.unpool4 = nn.MaxUnpool2d(2, stride=2, padding=0)
        self.deconv4_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.deconv4_3 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.unpool3 = nn.MaxUnpool2d(2, stride=2, padding=0)
        self.deconv3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.deconv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.deconv3_3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2, padding=0)
        self.deconv2_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.deconv2_2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.unpool1 = nn.MaxUnpool2d(2, stride=2, padding=0)
        self.deconv1_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.deconv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.output = nn.Conv2d(64, 21, 1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x, ind1 = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x, ind2 = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x, ind3 = self.pool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x, ind4 = self.pool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x, ind5 = self.pool5(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.deconv_fc6(x))
        x = torch.reshape(x, (x.shape[0], 512, 7, 7))
        x = self.unpool5(x, ind5)
        x = F.relu(self.deconv5_1(x))
        x = F.relu(self.deconv5_2(x))
        x = F.relu(self.deconv5_3(x))
        x = self.unpool4(x, ind4)
        x = F.relu(self.deconv4_1(x))
        x = F.relu(self.deconv4_2(x))
        x = F.relu(self.deconv4_3(x))
        x = self.unpool3(x, ind3)
        x = F.relu(self.deconv3_1(x))
        x = F.relu(self.deconv3_2(x))
        x = F.relu(self.deconv3_3(x))
        x = self.unpool2(x, ind2)
        x = F.relu(self.deconv2_1(x))
        x = F.relu(self.deconv2_2(x))
        x = self.unpool1(x, ind1)
        x = F.relu(self.deconv1_1(x))
        x = F.relu(self.deconv1_2(x))
        x = self.output(x)
        return x

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction) # look into this. It used to use label.squeeze()
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)
        
    def save_best_model(self, loss, file_path, num_to_keep=1):
        if self.test_loss == None or loss < self.test_loss:
            self.test_loss = loss
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)

    def accuracy(self, output, label):
        # print(output.shape)
        label = torch.tensor(img_util.rgb_to_onehot(label.cpu())).to(device)
        
        # print(label.shape)
        # output = output.numpy()
        # label = label.numpy()
        intersection = torch.logical_and(output, label)
        union = torch.logical_or(output, label)
        iou_score = torch.sum(intersection) / torch.sum(union)
        # print('IoU is %s' % iou_score)
        return iou_score
    
def predict(model, x):
    output = model.forward(x)
    prediction = torch.argmax(x, dim=1)
    return prediction