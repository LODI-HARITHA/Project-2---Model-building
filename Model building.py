####
!nvidia-smi

####
!pip install ultralytics

####
from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
!yolo mode=checks

####
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="eSY6Bks0HNmuOCGSdmrn")
project = rf.workspace("intelligent-bar-counting-nvsat").project("steel-bar-counting-team-a")
dataset = project.version(1).download("yolov8")

####
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=5 imgsz=416

####
!ls runs/detect/train/

####
Image(filename='/content/runs/detect/train6/confusion_matrix.png', width=600)

####
Image(filename='/content/runs/detect/train6/results.png', width=600)

####
Image(filename='/content/runs/detect/train6/val_batch0_pred.jpg', width=600)

#### Validate Custom Model
!yolo task=detect mode=val model=/content/runs/detect/train6/weights/best.pt data=/content/Steel-Bar-Counting-Team-A-1/data.yaml     

#### Inference with Custom Model
!yolo task=detect mode=predict model=/content/runs/detect/train6/weights/best.pt conf=0.25 source=/content/Steel-Bar-Counting-Team-A-1/test/images

####
import glob
from IPython.display import Image, display

for image_path in glob.glob('/content/runs/detect/predict/00038_jpg.rf.9d6158caad30fcead2f4cb87fe7ca826.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
     
