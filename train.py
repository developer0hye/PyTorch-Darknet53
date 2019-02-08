from model import *
from torchsummary import summary


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = darknet53(1000).to(device)
print(model)

summary(model.cuda(), (3, 416, 416))

