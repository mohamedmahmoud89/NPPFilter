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

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <filesystem>

bool printfNPPinfo(int argc, char *argv[])
{
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

void runFilter(const std::string &sFilename, const std::string &sResultFilename)
{
  // declare a host image object for an 8-bit grayscale image
  npp::ImageCPU_8u_C1 oHostSrc;
  // load gray-scale image from disk
  npp::loadImage(sFilename, oHostSrc);
  // declare a device image and copy construct from the host image,
  // i.e. upload host to device
  npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

  // create struct with box-filter mask size
  NppiSize oMaskSize = {5, 5};

  NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
  NppiPoint oSrcOffset = {0, 0};

  // create struct with ROI size
  NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
  // allocate device image of appropriately reduced size
  npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
  // set anchor point inside the mask to (oMaskSize.width / 2,
  // oMaskSize.height / 2) It should round down when odd
  NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

  // run box filter
  NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
      oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
      oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
      NPP_BORDER_REPLICATE));

  // declare a host image for the result
  npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
  // and copy the device result data into it
  oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

  saveImage(sResultFilename, oHostDst);
  std::cout << "Saved image: " << sResultFilename << std::endl;
}

int checkFileError(const std::string& sFilename)
{
  int file_errors = 0;
  std::ifstream infile(sFilename.data(), std::ifstream::in);

  if (infile.good())
  {
    std::cout << "boxFilterNPP opened: <" << sFilename.data()
              << "> successfully!" << std::endl;
    file_errors = 0;
    infile.close();
  }
  else
  {
    std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">"
              << std::endl;
    file_errors++;
    infile.close();
  }
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    std::string path = "data/";

    auto files = std::filesystem::directory_iterator(path);

    for (const auto &entry : files)
    {
      std::cout << "running filter on "<< entry.path() << std::endl;
      sFilename = entry.path();

      if (checkFileError(sFilename) > 0)
      {
        exit(EXIT_FAILURE);
      }

      const std::string sResultPath = "output/";
      std::string sResultFilename;

      std::string::size_type dot = sFilename.rfind('.');
      std::string::size_type slash = sFilename.rfind('/');

      if (dot != std::string::npos && slash != std::string::npos)
      {
        sResultFilename = sFilename.substr(slash + 1, dot - slash + 1);
      }

      sResultFilename = sResultPath + sResultFilename  + "_boxFilterOutput.pgm";

      runFilter(sFilename, sResultFilename);
    }

    printf("Finishing...\n");
    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
