# Darknet53

This is implementation of Darknet53 network discussed in [ [1] ](https://pjreddie.com/media/files/papers/YOLOv3.pdf) used for feature extractor of YOLOv3.

# Network Structure

![webp net-resizeimage](https://user-images.githubusercontent.com/35001605/53487913-2df2b480-3ad0-11e9-9788-b2feab624786.png)

# Trainining

- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use [the following script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
    
```
python train.py --data [your imagenet directory] --gpu 0 -b 64
```
b is a batch size for training and test, you can adjust this number.

## Reference
>[ [1] YOLOv3: An Incremental Improvement ](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
