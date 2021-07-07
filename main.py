import enum
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import warnings
# this is to supress bug warning from current version of pytorch
# check here for details: https://github.com/pytorch/pytorch/issues/54846
warnings.filterwarnings("ignore")


class VectorModels(enum.Enum):
    ResNet18 = 1
    MobileNetV2 = 2
    MobileNetV3 = 3


class Vectorizer:
    def __init__(self, cuda=False, model_name=VectorModels.ResNet18):
        # initialize image preprocessor for converting image binary
        # to format that match the input for vectorizer model
        self.image_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        # determine model that we will be using for the vectorizer
        model = None
        if model_name is VectorModels.ResNet18:
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name is VectorModels.MobileNetV2:
            model = models.mobilenet_v2(pretrained=True)
            model = model._modules.get('features')
        elif model_name is VectorModels.MobileNetV3:
            model = models.mobilenet_v3_small(pretrained=True)
            model = model._modules.get('features')
        # set the model mode into evaluation mode, this mode is used
        # when the model is used for inferencing
        model.eval()
        # if cuda parameter is enabled but cuda is not available
        # raise exception
        if cuda and not torch.cuda.is_available():
            raise ValueError("Cuda is not available")
        # save cuda flag
        self.cuda = cuda
        if self.cuda:
            model.to('cuda')
        # set the model to property
        self.model = model

    def get_vector(self, img):
        # convert image binary into input tensor
        input_batch = self.image_preprocess(img).unsqueeze(0)
        if self.cuda:
            input_batch = input_batch.to('cuda')
        return self.model(input_batch).flatten().detach().numpy()


v = Vectorizer(model_name=VectorModels.ResNet18)
input_image = Image.open('face_2.jpg')
compare_images = [
    Image.open('car_1.jpg'),
    Image.open('car_2.jpg'),
    Image.open('car_3.jpg'),
    Image.open('cat_1.jpg'),
    Image.open('cat_2.jpg'),
    Image.open('catdog_1.jpg'),
    Image.open('face_1.jpg'),
    Image.open('face_2.jpg'),
]

for compare_image in compare_images:
    vector_input = v.get_vector(input_image)
    vector_compare = v.get_vector(compare_image)

    dist = np.linalg.norm(vector_input - vector_compare)
    print(dist)
