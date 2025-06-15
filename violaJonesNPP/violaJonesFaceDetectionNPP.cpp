/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <getopt.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <sys/stat.h>
#include <sys/types.h>

// OpenCV
#include <opencv2/opencv.hpp>

bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

bool faceDetectionToFile(int argc, char *argv[], std::string sFilename) {
  try {
    int opt;
    int rc;

    // Variables to store start and end times for performance measurement.
    struct timespec t_start;
    struct timespec t_end;

    // Strings for input and output file paths.
    std::string sResultFilename;
    std::string sCascadeFilename;

    // Parse command-line options using getopt.
    while ((opt = getopt(argc, argv, "i:o:f:")) != -1) {
      switch (opt) {
      case 'i':
        if (access(optarg, F_OK) != 0) {
          std::cerr << "[ERROR] Input file " << optarg << " does not exist." << std::endl;
          exit(1);
        }
        sFilename = optarg;
        std::cout << "Input directory: <" << sFilename.data() << std::endl;
        break;
      case 'o':
        sResultFilename = optarg;
        std::cout << "Output directory: <" << sResultFilename.data() << std::endl;
        break;
      case 'f':
        sCascadeFilename = optarg;
        std::cout << "Cascade filter directory: <" << sCascadeFilename.data() << std::endl;
        break;
      default:
        std::cerr << "[ERROR] Incorrect parameter format.\n";
        exit(1);
      }
    }

    // Load the classifier
    std::cout << "Loading cascade classifier..." << std::endl;
    cv::CascadeClassifier classifier(sCascadeFilename);
    if (classifier.empty()) {
      std::cerr << "[ERROR] Failed to load cascade classifier from " << sCascadeFilename << std::endl;
      return false;
    }
    std::cout << "Cascade classifier loaded..." << std::endl;

    // Load grayscale image from disk into host
    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage(sFilename, oHostSrc);

    // Upload to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // Copy device image back to host for OpenCV processing
    npp::ImageCPU_8u_C1 oHostGray(oDeviceSrc.size());
    oDeviceSrc.copyTo(oHostGray.data(), oHostGray.pitch());

    // Wrap host gray image in cv::Mat
    cv::Mat grayMat(oHostGray.height(), oHostGray.width(), CV_8UC1, oHostGray.data());
    std::cout << "Grayscale image loaded successfully." << std::endl;
    std::cout << "Grayscale image dimensions: " << oHostGray.width() << " x " << oHostGray.height() << std::endl;

    // Convert to 3-channel for drawing rectangles
    cv::Mat imageBGR;
    cv::cvtColor(grayMat, imageBGR, cv::COLOR_GRAY2BGR);
    
    // Start the timer for performance measurement
    rc = clock_gettime(CLOCK_REALTIME, &t_start);
    assert(rc == 0);

    // Detect features
    std::vector<cv::Rect> features;
    classifier.detectMultiScale(grayMat, features, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    std::cout << "Face detection found " << features.size() << " results." << std::endl;
    
    // Stop the timer and calculate the runtime
    rc = clock_gettime(CLOCK_REALTIME, &t_end);
    assert(rc == 0);

    unsigned long long int runtime = 1000000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_nsec - t_start.tv_nsec;

    // Output the number of results and runtime details
    std::cout << "Time spent on detection: " << runtime / 1000000000 << "." << runtime % 1000000000 << " seconds (" << runtime << " nanoseconds)" << std::endl;

    // Draw face detection rectangles and log rectangle positions
    int i = 0;
    for (const auto &rect : features) {
      cv::rectangle(imageBGR, rect, cv::Scalar(255, 255, 255), 2);
      std::cout << "Face detection result #" << i + 1 << ": x=" << rect.x << ", y=" << rect.y << ", width=" << rect.width << ", height=" << rect.height << std::endl;
      i++;
    }

    // Convert back to grayscale for output
    cv::Mat finalGray;
    cv::cvtColor(imageBGR, finalGray, cv::COLOR_BGR2GRAY);

    // Copy to NPP host image
    npp::ImageCPU_8u_C1 oHostDst(finalGray.cols, finalGray.rows);
    std::memcpy(oHostDst.data(), finalGray.data, finalGray.total());

    // Save output
    std::string::size_type slash = sResultFilename.rfind('/');
    if (slash != std::string::npos) {
      std::string outputDir = sResultFilename.substr(0, slash);
      struct stat st;
      if (stat(outputDir.c_str(), &st) != 0 && mkdir(outputDir.c_str(), 0755) != 0)
      {
        std::cerr << "Failed to create output directory: " << outputDir << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    std::cout << "Saving output to: " << sResultFilename << std::endl;
    saveImage(sResultFilename, oHostDst);
    std::cout << "Image saved as: " << sResultFilename << std::endl;

    // Clean up device memory
    nppiFree(oDeviceSrc.data());
  }
  catch (...)
  {
    std::cerr << "[ERROR] Filtering error! An unknown exception occurred." << std::endl;
    ;
    exit(EXIT_FAILURE);
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  std::cout << argv[0] << " Starting...." << std::endl;

  try {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    } else {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath) {
      sFilename = filePath;
    } else {
      sFilename = "Lena.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good()) {
      std::cout << "imagefaceDetectionNPP opened: <" << sFilename.data()
                << "> successfully." << std::endl;
      file_errors = 0;
      infile.close();
    } else {
      std::cout << "imagefaceDetectionNPP unable to open: <" << sFilename.data() << ">."
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0) {
      exit(EXIT_FAILURE);
    }

    bool isFiltered = faceDetectionToFile(argc, argv, sFilename);

    if (isFiltered) {
      exit(EXIT_SUCCESS);
    } else {
      exit(EXIT_FAILURE);
    }
  } catch (npp::Exception &rException) {
    std::cerr << "[ERROR] Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "[ERROR] Program error! An unknown type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}