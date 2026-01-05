# README

We have multiple approaches for Image Segmentation using Ant-Colony Optimization (ACO). Each algorithm looks at the problem from a different angle.

There are some sample images in the `/images` folder. These images come from the "Oxford-IIIT Pet Dataset".

## 1. Image Thresholding

## 2. Edge Detection ACO

We use ACO to detect the edges in an image. After we find the edges, we need to find a way to actually segment the image into objects. We do this using a Watershed algorithm approach. The images are first transformed into grayscale as the algorithm only looks at the intensity of the pixels. To run the code, you need to run:

`python3 gray_edge_aco.py`

## 3. Color Edge Detection ACO

The issue with the previous approach is that it loses the information from the RGB channels and for example, for the image Abyssnian_106.jpg, we can clearly see the cat is on a red background, but after applying grayscale, the difference between brown and red is lost and the previous approach fails. This is why we looked at the color channels next.