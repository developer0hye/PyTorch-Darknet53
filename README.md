# Darknet53

This is implementation of Darknet53 network discussed in [ [1] ](https://pjreddie.com/media/files/papers/YOLOv3.pdf) used for feature extractor of YOLOv3.

This new network is more efficient than ResNet-101 or ResNet-152.
Here are some ImageNet results:

![darknet_table](https://user-images.githubusercontent.com/35001605/53488653-4b288280-3ad2-11e9-9aba-f14cbfc65c0c.PNG)

Each network is trained with identical settings and tested at 256×256, single crop accuracy. 

Run times are measured on a Titan X at 256 × 256. 

Thus Darknet-53 performs on par with state-of-the-art classifiers but with fewer floating point operations and more speed. 

**Darknet-53 is better than ResNet-101 and 1.5× faster.**

**Darknet-53 has similar performance to ResNet-152 and is 2× faster.** [ [1] ](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

This experiment used the Darknet neural network framework for training and testing [ [2] ](https://github.com/pjreddie/darknet).

## Network Structure

![webp net-resizeimage](https://user-images.githubusercontent.com/35001605/53487913-2df2b480-3ad0-11e9-9788-b2feab624786.png)

## Trainining

- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use [the following script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)


```
```

```
python train.py --data [your imagenet directory] --gpu 0 -b 64
```
b is a batch size for training and test, you can adjust this number.

## Benchmark
- framework: PyTorch
- GPU: GTX 1080 Ti 11GB
- Batchsize: 1
- Input shape: 3 x 224 x 224

**GPU time**
```
resnet101 : 0.034906
resnet152 : 0.055852
densenet121 : 0.041888
darknet53 : 0.017952
```

**CPU time**
```
resnet101 : 0.675194
 resnet152 : 0.949459
densenet121 : 0.649266
 darknet53 : 0.405916
```

## Reference
>[ [1] YOLOv3: An Incremental Improvement ](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
>[ [2] darknet framework ](https://github.com/pjreddie/darknet)
