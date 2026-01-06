# README

We have multiple approaches for Image Segmentation using Ant-Colony Optimization (ACO). Each algorithm looks at the problem from a different angle.

There are some sample images in the `/images` folder. These images come from the "Oxford-IIIT Pet Dataset".

If you want to visualize the results and you want to change the image, then you need to enter that script and change manually in the code the image path. For training the NN model we used the data splits obtained from running split_dataset.py. We used only the Abyssinian cats from the dataset.

## 1. Image Thresholding

To use image thresholding, you need to run:

`python3 image_threshold.py`

## 2. Edge Detection ACO

We use ACO to detect the edges in an image. After we find the edges, we need to find a way to actually segment the image into objects. We do this using a Watershed algorithm approach. The images are first transformed into grayscale as the algorithm only looks at the intensity of the pixels. To run the code, you need to run:

`python3 gray_edge_aco.py`

## 3. Color Edge Detection ACO

The issue with the previous approach is that it loses the information from the RGB channels and for example, for the image Abyssnian_106.jpg, we can clearly see the cat is on a red background, but after applying grayscale, the difference between brown and red is lost and the previous approach fails. This is why we looked at the color channels next.

# Evaluation

To evaluate numerically the pipelines, we created the evaluate.py script. First, you need to run each script to create the files with the masks (each script can be run on a whole folder with images, or individually on a certain image to see how it looks qualitatively). After you create the mask files, you will be able to evaluate. The Oxford-IIIT Pet dataset contains foreground and background. Our edge detectors detect multiple objects, so we decide that an object is in the foreground if it overlaps with a foreground object.