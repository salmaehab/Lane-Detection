# Lane-Detection

## Table of Contents
- Introduction
- Technologies Used
- Contributors
- Installation
- User Guide

## Introduction
The project aims:
- Identify the lane boundaries in a video from a front-facing camera on a car.
- Find and track the lane linesand the position of the car from the center of the lane. As a bonus, you can track the radius of curvature of the road too.

## Technologies Used
- Python Programming Language.
- numpy, matplotlib, opencv libraries.

## Contributors
[Salma Ehab](https://github.com/salmaehab), [Salma Hamed](https://github.com/Salma-Hamed), [Salma Hamdy](https://github.com/salma39), [Antoine Zakaria](https://github.com/AntoineZakaria)

## Installation Instructions
From (git bash) run the command: bash script.sh video input_path output_path mode

## User Guide
- To process a video set the first argument to 1, and to process an image set it to 0 
- The file has two modes of execution:
    - 0: for debugging
        This mode selects a frame from the input video and processes it, then it concatenates the output image from each stage and the saves the image.
    - 1: for running
