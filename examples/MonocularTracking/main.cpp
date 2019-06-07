#include "iostream"
#include "FeatureTracker.h"

int main(int argc, char *argv[]) {
    // Set up a monocular feature tracker
    FeatureTracker ft = FeatureTracker(MODE::MONO);
    
    std::cout << "Hello library!" << std::endl;

}