# Image clustering based on transfer learning features

## How to use
1. Install the following packages
* Keras
* numpy
* pandas
* natsort
* scikit-learn
* Pillow

2. Data preparation
* create a folder "img all" and place the images inside it. Please convert all your images to .jpg format first.
* create a txt file that has all the image filenames, seperated by each line.

3. Run the following Python scripts

## 1 feature extraction.py
For each image, this script extracts features from a pre-trained model and saves features in the "img exfeature" folder. If you want to use the hybrid model, please download the model from this [link](https://github.com/GKalliatakis/Keras-VGG16-places365) and place all the relevant scripts in the same folder. The script can also extract features from VGG16 model provided in the Keras package. 

## 2 combine features.py
This script combines all the features into one file.

## 3 PCA.py
This script conducts principal component analysis on the extract features.

## 4 kmeans clustering.py
This script applies k-means clustering to the first 200 dimensions in PCA, with the number of clusters ranging from 5 to 20.

## 5 copy image.py
For each cluster in each clustering solution, this script randomly selects 20 images and copies them to the "img cluster" folder.

## 6 visualize grid.py
For each clustering solution, this script creates a figure that show the randomly selected 20 images in each cluster.

If you have used the scripts, please cite the following article, which has a detailed description of the method:

Peng, Y. (2020). What Makes Politicians’ Instagram Posts Popular? Analyzing Social Media Strategies of Candidates and Office Holders with Computer Vision. The International Journal of Press/Politics.

```
@article{peng2021makes,
  title={What Makes Politicians’ Instagram Posts Popular? Analyzing Social Media Strategies of Candidates and Office Holders with Computer Vision},
  author={Peng, Yilang},
  journal={The International Journal of Press/Politics},
  volume={26},
  number={1},
  pages={143--166},
  year={2021},
  publisher={SAGE Publications Sage CA: Los Angeles, CA}
}
```

