//
//  ImageProcessing.h
//  GeometryAnimationFinal
//
//  Created by Nnenna EC Okwara on 12/9/12.
//  Copyright (c) 2012 Nnenna EC Okwara. All rights reserved.
//

#ifndef __GeometryAnimationFinal__ImageProcessing__
#define __GeometryAnimationFinal__ImageProcessing__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

using namespace std;
using namespace cv;
using namespace Eigen;

typedef Triplet<float> T;

void histogramMatching(const Mat input, const Mat matching, Mat &output);
class Image{
    float sigmaS, sigmaR;
    Mat colorIm;
public:
    Mat mat;
    Image(Mat);
    Image(Mat, int);
    float getSigmaS() {return sigmaS;}
    float getSigmaR() {return sigmaR;}
    Mat bf(Image I2);
    Mat highPassFilter();
    void setSigmaS(float s);
    void displayColor(Mat ending);
};
//Mat bf(const Mat I1, const Mat I2, float sigmaS, float sigmaR);
//Mat highPassFilter(Mat I, float sigma);
void absolute(Mat &mat);
void historgramTransfer(Mat I, Mat M, Mat &O);
void intensityTransfer(Mat I, Mat M, Mat &O);
float vTau(float x);
float grad90(Mat image);
int insertCoefficients(int i, int j, int pixC, int val, int rows, int cols, vector<T> &nonZeros, Eigen::VectorXf &b, float v1, float v2);
Mat poissonReconstruction(Mat img, Mat Gx, Mat Gy);

#endif /* defined(__GeometryAnimationFinal__ImageProcessing__) */
