# Face-datection : Image Segmentation Preprocessing Pipeline
This repository provides a preprocessing pipeline for supervised image segmentation tasks. It converts annotated RGB images with bounding boxes into training-ready datasets containing input images and corresponding binary masks. The processed data can be directly used for training neural networks such as U-Net or other segmentation architectures.

## Features

- Load and visualize annotated images
- Automatically generate object masks from bounding box annotations
- Resize images and masks to a consistent size (100x50 by default)
- Save preprocessed data as `.npz` files
- Overlay masks on images for visual inspection
- Normalize data for deep learning models
- Split data into training and validation sets

---
## Technologies Used

- Python 3.x
- NumPy, Matplotlib, OpenCV, PIL – image processing
- scikit-learn – data splitting
- Keras (TensorFlow backend) – model training (future work)
- tqdm – progress bars

---
## Dataset Format

Each entry in the dataset is a tuple:

```python
(image_array, annotations)
````

Where:

* `image_array` is a NumPy RGB image
* `annotations` is a list of bounding boxes in the following format:

```json
{
  "points": [
    { "x": float, "y": float },  // top-left corner
    { "x": float, "y": float }   // bottom-right corner
  ]
}
```

---

## Pipeline Overview

1. Load raw `.npy` dataset
2. Create binary masks for bounding boxes
3. Resize images and masks to standard size
4. Stack and save processed data in `.npz` format
5. Visualize samples for quality assurance
6. Normalize and split into training/validation sets

---

## Output

You will get files such as:

```
training_data_processed4.npz
training_data_processed5.npz
```

Each contains:

* `train` – processed RGB images
* `train_labels` – binary masks

These files are merged and used for training neural networks.

---

## Sample Output

Example visualization (image with overlayed mask):

```python
plt.imshow(image)
plt.imshow(mask, cmap="gray", alpha=0.5)  # overlay
```

To display sample results, you can insert a visualization image in your repository and refer to it here.

---

## Future Work

* Add support for polygon or multi-class masks
* Implement a full training pipeline (e.g., using U-Net)
* Add prediction and evaluation tools
---

(If the code is not yet modularized, consider breaking it into standalone scripts.)

## Contributions

Issues and pull requests are welcome.

