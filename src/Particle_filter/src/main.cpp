#include <iostream>

#include <math.h>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include "Particle.hpp"


int main()
{
    Motion mo;
    mo.runloop();
    return 0;
}