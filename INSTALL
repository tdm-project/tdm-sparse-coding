TDM_SPARSE_CODING Installation
==============================

This page describes how to build and install tdm_sparse_coding.

The tdm_sparse_coding build process requires CMake version 3.0 or higher and a
working compiler. It also requires make.

The installation procedure itself is broken into four steps.

   1. Dependencies
   2. Configuring
   3. Building
   4. Installing 

1. Dependencies
-------------------------

TDM_SPARSE_CODING depends on the following libraries:

sl: C++ library that support geometric processing developed by CRS4. Available on http://vic.crs4.it/vic/download/. License: Free for non commercial use.

Eigen: C++ template linear algebray library. Available on http://eigen.tuxfamily.org/index.php?title=Main_Page. License: Mozilla  PublicLicense 2.0. Free/Copyleft

You must download the dependencies and install them following the instructions.

In addition,the library and program are configured and built using the cmake tool:

cmake 
Homepage: http://www.cmake.org/ 
Description: Cross platform Make (with make backend) 

We also use a c++ compiler. While many solutions are available, the code has been tested using GCC on a Linux platform.

Make sure that all prerequisites are installed and functional before proceeding.

2. Configuring
--------------

The code must be build in a build directory that you need to create before building the code. 
In the following we use: "srcdir" to refer to the toplevel source
directory for TDM_SPARSE_CODING; "objdir" to refer to the build directory.

You can configure TDM_SPARSE_CODING via cmake:

  % cd objdir
  % cmake [options] srcdir

  where options can include:

    -DCMAKE_INSTALL_PREFIX=PREFIX

      install architecture-independent files in PREFIX
      [default is /usr/local] 

    -DCMAKE_BUILD_TYPE=TYPE (only for Linux)

      control the type of build when using a single-configuration
      generator like the Makefile generator. CMake will create by
      default the following variables when using a
      single-configuration generator:

      * None (CMAKE_C_FLAGS or CMAKE_CXX_FLAGS used)
      * Debug (CMAKE_C_FLAGS_DEBUG or CMAKE_CXX_FLAGS_DEBUG)
      * Release (CMAKE_C_FLAGS_RELEASE or CMAKE_CXX_FLAGS_RELEASE)
      * RelWithDebInfo (CMAKE_C_FLAGS_RELWITHDEBINFO or 
                        CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      * MinSizeRel (CMAKE_C_FLAGS_MINSIZEREL or 
                    CMAKE_CXX_FLAGS_MINSIZEREL) 

      You can use these default compilation flags (or modify them) by
      setting the CMAKE_BUILD_TYPE variable at configuration time from
      within the "ccmake" GUI. Note! 

      If you are using the Makefile generator, you can create your own
      build type like this:

        SET(CMAKE_BUILD_TYPE distribution)
        SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3") 
        SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3") 
      
      [default is "Release"]

Other useful cmake variables are here:
http://www.cmake.org/Wiki/CMake_Useful_Variables 

3. Building
-----------

Now that TDM_SPARSE_CODING is configured, you are ready to build the TDM_SPARSE_CODING library and test program.

To build the library, simply type:

  % cd objdir; make


4. Installing
-------------

Now that TDM_SPARSE_CODING has been built, you can install it with:

  % cd objdir; make install

Enjoy.
