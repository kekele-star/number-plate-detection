<a href="https://colab.research.google.com/github/kekele-star/number-plate-detection/blob/main/train-set.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Topic: Introduction to Object Detection**

# **Content**

1. [Overview of Object Detection](#part1)
  - 1.1 [Key Components of Object Detection Models](#part1.1)
  - 1.2 [ Popular Object Detection Models](#part1.2)
  - 1.3 [Tools and Libraries](#part1.3)

2. [Data Prepapration & Exploration](#part2)
  - 2.1 [Data Formats](#part2.1)
  - 2.2 [Data Annotation](#part2.2)

3. [Inference with Pre-trained Models](#part3)

4. [Evaluation Metrics](#part4)
  - 4.1 [Common Metrics](#part4.1)

5. [Finetuning a Pretrained Model on License Plate Dataset](#part5)


<a name="part1"></a>

#  Overview of Object Detection
Object detection is a computer vision technique that involves identifying and locating objects within an image or video. Unlike image classification, which assigns a single label to an image, object detection provides both the label and the coordinates of bounding boxes for each object. Popular applications include autonomous driving, security surveillance, and image search engines.
 <div align="center">
  <img src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/68/ae/object-detection-figure-1.component.complex-narrative-xl.ts=1713813010967.jpg/content/adobe-cms/us/en/topics/object-detection/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative/items/content_group_745532756/image" alt="Labeling GIF" width="800" height="800">
</div>

<a name="part1.1"></a>
## Key Components of Object Detection Models

### 1. Feature Extraction
- **Convolutional Neural Networks (CNNs):** CNNs are the backbone of many object detection models. They are used to extract features from images through convolutional layers, pooling layers, and activation functions.
- **Pre-trained Models:** Using models pre-trained on large datasets (e.g., ImageNet) helps in leveraging learned features and reduces training time.

### 2. Region Proposal
- **Selective Search:** An algorithm that proposes possible object regions by combining similar regions based on color, texture, size, and shape.
- **Region Proposal Networks (RPNs):** Networks that generate object proposals directly from feature maps, commonly used in models like Faster R-CNN.

### 3. Classification and Localization
- **Bounding Box Regression:** A technique used to predict the coordinates of the bounding box around detected objects.
- **Object Classification:** Assigning a label to each detected object using fully connected layers and softmax activation.

### 4. Anchor Boxes
Pre-defined bounding boxes of different sizes and aspect ratios used to detect objects at various scales and shapes. Commonly utilized in models like YOLO and SSD.

### 5. Loss Functions
- **Classification Loss:** Measures the accuracy of predicted object classes.
- **Localization Loss:** Measures the accuracy of predicted bounding box coordinates.
- **Total Loss:** A combination of classification and localization losses to optimize the model.

### 6. Post-Processing
- **Non-Maximum Suppression (NMS):** An algorithm used to remove redundant bounding boxes and keep only the most accurate ones.



<a name="part1.2"></a>
## Popular Object Detection Models

### 1. R-CNN (Regions with CNN features)
- **Architecture:** Extracts region proposals using Selective Search, then uses a CNN to extract features from each region, followed by classification using a linear SVM.
 <div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*yJBiHhK8t_zTQKBqlZKlWQ.png" alt="Labeling GIF" width="800" height="500">
</div>

### 2. Fast R-CNN
- **Architecture:** Improves upon R-CNN by sharing convolutional computations and using a single-stage training process. It uses Region of Interest (RoI) pooling to extract fixed-size feature maps for each region proposal.
 <div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*7haXXOJjZdibibU0c4eNgw.png" alt="Labeling GIF" width="800" height="500">
</div>

### 3. Faster R-CNN
- **Architecture:** Combines Region Proposal Networks (RPN) with Fast R-CNN. The RPN proposes regions, and the Fast R-CNN model classifies them and refines bounding boxes.
 <div align="center">
  <img src="https://www.researchgate.net/publication/326668850/figure/fig1/AS:653294032666626@1532768843320/Faster-R-CNN-architecture-Top-left-box-represents-the-base-network-box-on-the-right.png" alt="Labeling GIF" width="800" height="500">
</div>

### 4. YOLO (You Only Look Once)
- **Architecture:** Divides the image into a grid and predicts bounding boxes and probabilities for each grid cell. Known for its speed.
 <div align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/1512812.png" alt="Labeling GIF" width="800" height="500">
</div>

### 5. SSD (Single Shot MultiBox Detector)
- **Architecture:** Detects objects in images using a single deep neural network, combining predictions from multiple feature maps with different resolutions.
<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*La_I2VXlENAJ9r0Wpf_vMg.jpeg" alt="Labeling GIF" width="800" height="500">
</div>


<a name="part1.3"></a>
## Tools and Libraries
In this section, we introduce the essential libraries used for object detection:
- **OpenCV**: An open-source computer vision library that provides tools for image processing.
- **TensorFlow**: A deep learning framework developed by Google, commonly used for machine learning tasks.
- **PyTorch**: An open-source machine learning library developed by Facebook, known for its dynamic computation graph.
- **Detectron2**: A PyTorch-based library developed by Facebook AI Research for object detection and segmentation.


<a name="part2"></a>
# Data Preparation


<a name="part2.1"></a>

## Data Formats
There are several formats for object detection datasets, the most common being:
#### 1. COCO (Common Objects in Context)

COCO is one of the most widely used data formats in object detection, known for its rich annotations which include object segmentation, keypoint detection, and captioning. A typical COCO dataset includes:
- **Images**: A list of images with metadata like file names and sizes.
- **Annotations**: Each annotation includes an image ID, category ID, bounding box coordinates, and segmentation information.
- **Categories**: A list of categories with IDs and names.

**Example of a COCO annotation:**

```json
{
  "images": [
    {
      "id": 123,
      "width": 800,
      "height": 600,
      "file_name": "000000123.jpg"
    }
  ],
  "annotations": [
    {
      "id": 456,
      "image_id": 123,
      "category_id": 1,
      "bbox": [192, 78, 200, 300],
      "segmentation": RLE or [polygon],
      "area": 60000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person"
    }
  ]
}
```
*Note: The ```bbox``` field contains the coordinates as [x, y, width, height].*

---


#### 2. Pascal VOC (PASCAL Visual Object Classes)
Pascal VOC is another popular format, particularly for earlier object detection challenges. It uses XML files to store annotation information for each image, including object class labels and bounding box coordinates.

**Example of a Pascal VOC annotation:**
```json
<annotation>
    <folder>VOC2012</folder>
    <filename>000001.jpg</filename>
    <size>
        <width>800</width>
        <height>600</height>
        <depth>3</depth>
    </size>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>174</xmin>
            <ymin>101</ymin>
            <xmax>349</xmax>
            <ymax>351</ymax>
        </bndbox>
    </object>
</annotation>
```
*Note: The ```bndbox``` contains the bounding box coordinates as [xmin, ymin, xmax, ymax].*

---
#### 4. Plain Text (.txt) Format (Used by YOLO)
In YOLO (You Only Look Once), annotations are stored in plain text (.txt) files. Each image has a corresponding .txt file where each line represents one object in the image. The format typically includes the class ID and normalized bounding box coordinates (center x, center y, width, height).

**Example of a YOLO .txt annotation:**
```
0 0.4921875 0.3958333 0.25390625 0.375
1 0.244140625 0.4708333 0.140625 0.175
```
---
#### 4. Custom datasets
For projects that do not fit the standard datasets, custom annotations can be made. These might follow similar structures but often require additional setup or preprocessing to ensure compatibility with object detection models.


<a name="part2.2"></a>

##  Data Annotation
Data annotation is the process of labeling images with bounding boxes and class labels. Tools used for data annotaion includes
 and are popular for this task. This
*   **LabelImg**
*   **Computer Vision Annotation Tool (CVAT)**
*   **VGG Image Annotator (VIA)**
*   **Roboflow**

<div align="center">
  <img src="https://blog.roboflow.com/content/images/2020/12/labeling.small-1.gif" alt="Labeling GIF" width="600" height="400">
</div>
