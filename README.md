# General Invariant Feature Tracker (GIFT)

GIFT is an image feature tracking application for common camera set-ups such as monocular, stereo, and multi-view.
The goal is to provide a package which simplifies the process of obtaining and tracking features from a sequence of images.

## Dependencies

Currently the GIFT depends on both Eigen 3 and OpenCV 3.
In the future the dependency on Eigen may be removed.

- Eigen3:  `sudo apt install libeigen3-dev`
- OpenCV:  `sudo apt install libopencv-dev`

## Building and Installing

GIFT can be built and installed using cmake and make.

```bash
git clone https://github.com/pvangoor/GIFT
cd GIFT
mkdir build
cd build
cmake ..
sudo make install
```
