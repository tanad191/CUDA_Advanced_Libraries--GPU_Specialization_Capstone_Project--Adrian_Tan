# CUDA_Advanced_Libraries--GPU_Specialization_Capstone_Project--Adrian_Tan
Capstone project repository for John Hopkins University's CUDA Advanced Libraries course on Coursera

## Overview

Proposed in 2001 by Paul Viola and Michael Jones, the **Viola–Jones object detection framework** was initially developed for face detection in videos and images, though its versatility and ease of implementation allow it to be used in a variety of other contexts. Its key component is a sequence of **classifiers** - a classifier is a single-layer neural network, or *perceptron*, with several binary masks or *Haar features* used for feature detection. A sliding window is computed over the image used as input and classifiers applied to the portion of the image captured by the window. A feature is considered detected if and only if all classifiers in the sequence output a positive result for their respective Haar features.

This implementation of the Viola-Jones framework is based on [a demonstration of the OpenCV library's object recognition functionality](https://gist.github.com/arrieta/3f701d9e461cc8228aae2540a71cdcae), as provided by Juan Arrieta, CEO of Nabla Zero Labs in South Pasadena, CA. While Arrieta's code employs objects and libraries from OpenCV, this project incorporates the NVIDIA Performance Primitives (NPP) library, which enables developers to perform GPU-accelerated image, video, and signal processing faster than CPU-only implementations. To this end, this project works with .pgm files as both input and output, storing them as NPP objects before converting them to OpenCV matrices for feature-detection and then converting the results back to NPP objects for output.

## Installing the Project

To build and run the code for this project, you will need to have the following installed:

1.  **GNU C++ Compiler** for building the application.

2.  **`make` Utility** for running the build process from the project terminal. Verify that this is installed and in your PATH by opening a command prompt or PowerShell and typing `make --version`. Version information should be displayed should `make` make be installed.

3.  OPTIONAL: **IrfanView (or another PGM viewer)** for viewing the PGM images that are both used as input and produced as output. You can also use it to convert JPEG/PNG images to the PGM format and vice versa. Download and install this from [https://www.irfanview.com/](https://www.irfanview.com/).


## Building and Running the Project

Once the above prerequisites are installed, navigate to the project directory:

Then use the provided `Makefile` to build the project with the following command:

```bash
$ make clean build
```

This will remove all binary ISO files and executables as applicable and then (re)create an executable file named `violaJonesFacialDetectionNPP`.

Once the project has been built and the executable generated, run the following commands to test the input:

```bash
$ make test_sample
$ make test_photography
$ make test_anime
```

*make test_sample* runs the executable with a single image, LennaTestImage.pgm, based on a standard test image used in digital image processing portraying the Swedish model Lena Forsén, shot by photographer Dwight Hooker and cropped from the centerfold of the November 1972 issue of Playboy magazine.

*make test_photography* and *make test_anime* are for the purposes of the project demonstration, and each command runs the executable file with eight different images within one subfolder within the /data directory. Comparisons can be made both between photographs of live actors from various movies and screencaps of various animated works (Japanese anime was used for the latter in this example), and between the inputs within either category to assess how the algorithm fares in different lighting conditions, with different numbers of people, and with different facial positions and orientations.