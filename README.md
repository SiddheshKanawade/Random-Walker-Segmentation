# Random-Walker-Segmentation

## Set Local Environment:
1. Install python>=3.8
2. run similar command `python randomwalker.py --imagePath "dataset/7.png" --totalSegments 2 --totalPixels 32`
3. A window should prompt up asking user to annotate the segments
4. After completion, one should get following result:
```terminal
[2022-11-23 11:23:17,785] [create-segmentation] [INFO] Storing Images at data
[2022-11-23 11:23:17,787] [create-segmentation] [INFO] Completed the segmentation
[2022-11-23 11:23:17,787] [create-segmentation] [INFO] 
[2022-11-23 11:23:17,787] [create-segmentation] [INFO] Total Segments used: 3
[2022-11-23 11:23:17,787] [create-segmentation] [INFO] Total Pixels used: 30
[2022-11-23 11:23:17,787] [create-segmentation] [INFO] Image Used: 0.png
[2022-11-23 11:23:17,787] [create-segmentation] [INFO] Calculating Error
[2022-11-23 11:23:17,787] [create-segmentation] [INFO] The mean absolute error is 37.524739583333336
```

## Note
If some python dependencies are not installed then run `pip install {name-of-dependency}`
