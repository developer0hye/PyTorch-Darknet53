import time
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable

from model import darknet53

def speed(model, name):
    with torch.no_grad():
        model.eval()

        t0 = time.time()
        input = torch.rand(1,3,224, 224).cuda()
        input = Variable(input)
        t1 = time.time()

        model(input)

        avg_time = 0

        for i in range(0, 10):
            torch.cuda.synchronize()
            t2 = time.time()

            model(input)

            torch.cuda.synchronize()
            t3 = time.time()

            avg_time += t3 - t2

        avg_time /= 10.0

        print('%10s : %f' % (name, avg_time))


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
