# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jerry/program/cplus/opencvwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jerry/program/cplus/opencvwork/build

# Include any dependencies generated for this target.
include CMakeFiles/imageBasics.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/imageBasics.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imageBasics.dir/flags.make

CMakeFiles/imageBasics.dir/imageBasics.cpp.o: CMakeFiles/imageBasics.dir/flags.make
CMakeFiles/imageBasics.dir/imageBasics.cpp.o: ../imageBasics.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jerry/program/cplus/opencvwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imageBasics.dir/imageBasics.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/imageBasics.dir/imageBasics.cpp.o -c /home/jerry/program/cplus/opencvwork/imageBasics.cpp

CMakeFiles/imageBasics.dir/imageBasics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imageBasics.dir/imageBasics.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jerry/program/cplus/opencvwork/imageBasics.cpp > CMakeFiles/imageBasics.dir/imageBasics.cpp.i

CMakeFiles/imageBasics.dir/imageBasics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imageBasics.dir/imageBasics.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jerry/program/cplus/opencvwork/imageBasics.cpp -o CMakeFiles/imageBasics.dir/imageBasics.cpp.s

# Object files for target imageBasics
imageBasics_OBJECTS = \
"CMakeFiles/imageBasics.dir/imageBasics.cpp.o"

# External object files for target imageBasics
imageBasics_EXTERNAL_OBJECTS =

imageBasics: CMakeFiles/imageBasics.dir/imageBasics.cpp.o
imageBasics: CMakeFiles/imageBasics.dir/build.make
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_gapi.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_stitching.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_aruco.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_bgsegm.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_bioinspired.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_ccalib.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_dnn_objdetect.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_dpm.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_face.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_freetype.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_fuzzy.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_hfs.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_img_hash.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_line_descriptor.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_reg.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_rgbd.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_saliency.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_stereo.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_structured_light.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_superres.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_surface_matching.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_tracking.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_videostab.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_xfeatures2d.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_xobjdetect.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_xphoto.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_shape.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_datasets.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_plot.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_text.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_dnn.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_ml.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_phase_unwrapping.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_optflow.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_ximgproc.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_video.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_objdetect.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_calib3d.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_features2d.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_flann.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_highgui.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_videoio.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_imgcodecs.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_photo.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_imgproc.so.4.0.1
imageBasics: /home/jerry/program/cplus/opencv/build/lib/libopencv_core.so.4.0.1
imageBasics: CMakeFiles/imageBasics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jerry/program/cplus/opencvwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable imageBasics"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imageBasics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imageBasics.dir/build: imageBasics

.PHONY : CMakeFiles/imageBasics.dir/build

CMakeFiles/imageBasics.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imageBasics.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imageBasics.dir/clean

CMakeFiles/imageBasics.dir/depend:
	cd /home/jerry/program/cplus/opencvwork/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jerry/program/cplus/opencvwork /home/jerry/program/cplus/opencvwork /home/jerry/program/cplus/opencvwork/build /home/jerry/program/cplus/opencvwork/build /home/jerry/program/cplus/opencvwork/build/CMakeFiles/imageBasics.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imageBasics.dir/depend

