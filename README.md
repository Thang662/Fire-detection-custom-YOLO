# Custom yolo for fire detection

This is an repo for traning yolo with custom loss function to your defined loss function

## Tutorial
<details open>
<summary>Usage</summary>

### Set up environment and data

```bash
conda create -n yolo python==3.10.15 # Create with conda 
conda activate yolo

# Download data
#!/bin/bash
curl -L -o ./fire-detection-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/trnthngl/fire-detection-dataset
unzip fire-detection-dataset.zip

pip install -r requirments.txt
````

After that, create an `.env` file with your CometML API key as 

```
API_KEY = YOUR_API_KEY
```

### Train your model

```
python main.py --your-args
```

For `args` options, you can refer to `configs` folder with `model` config refer to `model name` and `loss function` and `train` for hyparameters for training in [YOLO](https://docs.ultralytics.com/modes/train/)


</details>

## [Yolov5](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/#43-eliminate-grid-sensitivity)

![Yolo anchor box](https://user-images.githubusercontent.com/31005897/158508119-fbb2e483-7b8c-4975-8e1f-f510d367f8ff.png#pic_center)

![Yolov5 bounding box](https://user-images.githubusercontent.com/31005897/158508027-8bf63c28-8290-467b-8a3e-4ad09235001a.png#pic_center)

- Predict the center of object based on offset acording to anchor template.

![alt text](https://user-images.githubusercontent.com/31005897/158508771-b6e7cab4-8de6-47f9-9abf-cdf14c275dfe.png#pic_center)
![](https://blog.roboflow.com/content/images/2024/04/image-1812.webp)

## [Yolov8](https://blog.roboflow.com/what-is-yolov8/)
![](https://blog.roboflow.com/content/images/2024/04/image-1816.webp)

