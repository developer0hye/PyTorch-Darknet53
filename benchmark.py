import time
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable

from model import darknet53

def speed(model, name):
    with torch.no_grad():
        torch.cuda.synchronize()
        input = torch.rand(1, 3, 224, 224).cuda()
        input = Variable(input)


        model(input)


        t1 = time.time()
        model(input)
        t2 = time.time()

        print('%10s : %f' % (name, t2 - t1))


if __name__ == '__main__':
    # cudnn.benchmark = True # This will make network slow ??
    resnet101 = models.resnet101().cuda()
    resnet152 = models.resnet152().cuda()
    densenet121 = models.densenet121().cuda()
    darknet = darknet53(1000).cuda()

    speed(resnet101, 'resnet101')
    speed(resnet152, 'resnet152')
    speed(densenet121, 'densenet121')
    speed(darknet, 'darknet53')