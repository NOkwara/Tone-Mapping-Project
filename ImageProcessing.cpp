 //
//  ImageProcessing.cpp
//  GeometryAnimationFinal
//
//  Created by Nnenna EC Okwara on 12/9/12.
//  Copyright (c) 2012 Nnenna EC Okwara. All rights reserved.
//

#include "ImageProcessing.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace Eigen;
extern const int channels = 256;
typedef Triplet<float> T;

/*
 Histogram mathcing for color images
 */
void histogramMatching(const Mat input, const Mat matching, Mat &output){
    
     float image_hist[channels] = {0};
    
    float image_ref[channels] = {0};
    
    double minVal, maxVal;
    minMaxLoc(input, &minVal, &maxVal);
    //double dist = maxVal - minVal; //values between max val and min val
    float fraction = (maxVal - minVal) / 256.0; // 255 is number of channel to split*/
    if(fraction < 0) fraction = 1;
	//create historgram fo red blue and green channels
	for(int x = 0; x < input.rows; x++) for(int y = 0; y < input.cols; y++){
        if(round(input.at<float>(x, y) / fraction) > 255) 
            image_hist[255]++;
        else if(round(input.at<float>(x, y) / fraction) < 0)
            image_hist[0]++;
        else{
            image_hist[(int) round(input.at<float>(x, y) / fraction)]++;
        }
		
	}
    
    for(int x = 0; x < matching.rows; x++) for(int y = 0; y < matching.cols; y++){
        if(round(matching.at<float>(x, y) / fraction) > 255)
            image_ref[255]++;
        else
            image_ref[(int) round(matching.at<float>(x, y) / fraction)]++;
	}
    //convert to cumulative histogram for each
    for(int i = 1; i < channels; i++){
        image_hist[i] += image_hist[i-1];
        image_ref[i] += image_ref[i-1];
    }
    //now divide to make the top 1
    for(int i = 0; i < channels; i++){
        image_hist[i] /= image_hist[channels - 1];
        image_ref[i] /= image_ref[channels - 1];
    }
    //now do histogram matching
    int point = 0;
    int image_match[channels];
    float num2match;
    for(int i = 0; i < channels; i++){
        num2match = image_hist[i];
        for(int j = point; j < channels; j++){
            if (num2match <= image_ref[j]){
                point = j;
                break;
            }
        }
    //This means for pixels in input with value i they should match to value point;
        image_match[i] = point;
    }
    //rebuild histogram using matching information
    for(int x = 0; x < output.rows; x++) for(int y = 0; y < output.cols; y++){
        if(round(input.at<float>(x, y) / fraction) > 255)
            output.at<float>(x, y) = (float) image_match[255] * fraction;
        else
            output.at<float>(x, y) = (float) image_match[(int) round(input.at<float>(x, y) / fraction)] * fraction;
	}
}

Image::Image(Mat image){
    image.convertTo(mat, CV_32F);
    mat /= 255;
       sigmaS = min(image.rows, image.cols)/16;
       sigmaR = grad90(image);
    
}

Image::Image(Mat image, int transformed){
    if(transformed){
        mat = image;
        sigmaS = min(image.rows, image.cols)/16;
        sigmaR = grad90(image);
    }
    else{
        image.convertTo(mat, CV_32F);
        mat /= 255;
        sigmaS = min(image.rows, image.cols)/16;
        sigmaR = grad90(image);
    }
    
}

float gaussian(float x, float sigma){
    return exp( (-x*x) / (sigma*sigma));
}

Mat Image::bf(const Image I2){
    
    //calculate K for each pixel
    Mat K;
    
    K = Mat::zeros(mat.rows, mat.cols, CV_32F);
    for(int n = 0; n < mat.rows; n++) for(int m = 0; m < mat.cols; m++){//for every element in K
        Point p(m, n);
        for(int i = n - 4; i < n + 4; i++) for(int j = m - 4; j < m + 4; j++){ //for every pixel in
            int x = i < 0? 0 : i;
            x = i >= I2.mat.rows? I2.mat.rows - 1 : x;
            int y = j < 0? 0 : j;
            y = j >= I2.mat.cols? I2.mat.cols - 1 : y;
            Point q(y, x);
            float Ip = mat.at<float>(n, m);
            float Iq = mat.at<float>(x, y);
            K.at<float>(n, m) += gaussian(norm(p - q), sigmaS) * gaussian(abs(Ip-Iq), sigmaR);
           // printf("K = %f x = %d y = %d\n", K.at<float>(n, m), n, m);
        }
    }
    
    Mat newImage(mat.rows, mat.cols, CV_32F);
    //now calculate bilateral filter
    for(int n = 0; n < mat.rows; n++) for(int m = 0; m < mat.cols; m++){//for every element in K
        Point p(m, n);
        float pixelV = 0;
        for(int i = n - 4; i < n + 4; i++) for(int j = m - 4; j < m + 4; j++){ //for every pixel in image
            int x = i < 0? 0 : i;
            x = i >= I2.mat.rows? I2.mat.rows - 1 : x;
            int y = j < 0? 0 : j;
            y = j >= I2.mat.cols? I2.mat.cols - 1 : y;
            Point q(y, x);
            float Ip = mat.at<float>(n, m);
            float Iq = mat.at<float>(x, y);
            pixelV += gaussian(norm(p - q), sigmaS) * gaussian(abs(Ip-Iq), sigmaR) * I2.mat.at<float>(x, y);
        }
        pixelV /= K.at<float>(n, m);
        newImage.at<float>(n, m) = pixelV;
        //printf("pix = %f, K = %f x = %d y = %d\n", pixelV, K.at<float>(m, n), n, m);
    }
    K.~Mat();
    return newImage;
}

Mat DFT(Mat *I, float cutoff){
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I->rows );
    int n = getOptimalDFTSize( I->cols ); // on the border add zero values
    
    copyMakeBorder(*I, padded, 0, m - I->rows, 0, n - I->cols, BORDER_CONSTANT, Scalar::all(0));
    
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I)
    
    int cx = planes[0].cols/2;
    int cy = planes[0].rows/2;
    
    Mat q0(planes[0], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(planes[0], Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(planes[0], Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(planes[0], Rect(cx, cy, cx, cy)); // Bottom-Right
    
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    q0.release();
    q1.release();
    q3.release();
    q2.release();
    
    Mat b0(planes[1], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat b1(planes[1], Rect(cx, 0, cx, cy));  // Top-Right
    Mat b2(planes[1], Rect(0, cy, cx, cy));  // Bottom-Left
    Mat b3(planes[1], Rect(cx, cy, cx, cy)); // Bottom-Right
                             // swap quadrants (Top-Left with Bottom-Right)
    b0.copyTo(tmp);
    b3.copyTo(b0);
    tmp.copyTo(b3);
    
    b1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    b2.copyTo(b1);
    tmp.copyTo(b2);
    
    int M = planes[0].cols; // x
    int N = planes[0].rows;// y
    
    //D(u, v) = sqrt((u - M/2)*(u - M/2) + (v - N/2)*(v - N/2))
    
    for(int v = 0; v < planes[0].rows; v++) for(int u = 0; u < planes[0].cols; u++){
        
        float H = 1 - exp((-1 * pow(sqrt((u - M/2)*(u - M/2) + (v - N/2)*(v - N/2)),2)) / (2 * cutoff * cutoff));
        planes[0].at<float>(v, u) *= H;
        planes[1].at<float>(v, u) *= H;
    }
    Mat q4(planes[0], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q5(planes[0], Rect(cx, 0, cx, cy));  // Top-Right
    Mat q6(planes[0], Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q7(planes[0], Rect(cx, cy, cx, cy)); // Bottom-Right
    
    q4.copyTo(tmp);
    q7.copyTo(q4);
    tmp.copyTo(q7);
    
    q5.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q6.copyTo(q5);
    tmp.copyTo(q6);
    
    Mat b4(planes[1], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat b5(planes[1], Rect(cx, 0, cx, cy));  // Top-Right
    Mat b6(planes[1], Rect(0, cy, cx, cy));  // Bottom-Left
    Mat b7(planes[1], Rect(cx, cy, cx, cy)); // Bottom-Right
    
    b4.copyTo(tmp);
    b7.copyTo(b4);
    tmp.copyTo(b7);
    
    b5.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    b6.copyTo(b5);
    tmp.copyTo(b6);
    
    //imshow("real returned", planes[0]);
    
    Mat magI = planes[0];
    
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE, complexI.rows);
    split(complexI, planes);
    
    magnitude(planes[0], planes[0], planes[1]);
    Mat mag = planes[0];
    mag += Scalar::all(1);
    
    log(mag, mag);
    
    
    mag = mag(Rect(0, 0, I->cols, I->rows));
    double minVal, maxVal;
    minMaxLoc(mag, &minVal, &maxVal);
    Mat draw;
    mag.convertTo(draw, CV_8U, 255/(maxVal - minVal), -minVal * 255 / (maxVal - minVal));
    draw.convertTo(mag, CV_32F);
    mag /= 255;
    return mag;
}

Mat Image::highPassFilter(){
    
    /*for(int u = 0; u < M; u++) for(int v = 0; v < N; v++){
     float duv = sqrt((u - (M/2.0)) * (u - (M/2.0)) + (v - (N/2.0)) * (v - (N/2.0)));
     fourierTransform.at<float>(v, u) = (1 - exp((-1 *duv*duv)/(2*625))) * fourierTransform.at<float>(v, u);
     }*/
    
        Mat H = DFT(&mat, sigmaS);
    
    return H;
}

void Image::setSigmaS(float s){
    sigmaS = s;
}

void absolute(Mat &mat){
    for(int i = 0; i < mat.cols; i++) for(int j = 0; j < mat.rows; j++){
        mat.at<float>(j, i) = abs(mat.at<float>(j, i));
    }
}

void historgramTransfer(Mat I, Mat M, Mat &O){
    
}

void intensityTransfer(Mat I, Mat M, Mat &O){
    
}

float vTau(float x){
    float tau = 0.1;
    if(x < tau)
        return 0;
    else if(x > 2*tau)
        return 1;
    else
        return 1 - pow(1 - (((x - tau)*(x-tau)) / (tau*tau)), 2);
}

float grad90(Mat image){
    //return 90th percentile gradient value
    Mat dx = Mat::Mat(1, 2, CV_32F);
    dx.at<float>(0,0) = 1; dx.at<float>(0,1) = -1;
    
    //Mat dy = Mat::Mat(2, 1, CV_32F);
    //dy.at<float>(0,0) = 1; dy.at<float>(1,0) = -1;
    
    Mat Idx;//, Idy;
    
    filter2D(image, Idx, image.depth(), dx);
    //filter2D(image, Idy, image.depth(), dy);
    
    float image_hist[channels] = {0};
    
	//create historgram fo red blue and green channels
	for(int x = 0; x < image.rows; x++) for(int y = 0; y < image.cols; y++){
        //printf("image value = %d", Idx.at<uchar>(x, y));
		image_hist[(int) Idx.at<uchar>(x, y)]++;
    }
	
    int i = 0;
    //convert to cumulative histogram for each
    for(i = 1; i < channels; i++)
        image_hist[i] += image_hist[i-1];
    
    //normalize
    for(i = 1; i < channels; i++)
        image_hist[i] /= image_hist[channels - 1];
    
    //get and return 90% value
    
    for(i = 1; i < channels; i++)
        if(image_hist[i] > 0.9)
            break;
    
    return i;
}

int insertCoefficients(int i, int j, int pixC, int val, int rows, int cols, vector<T> &nonZeros, Eigen::VectorXf &b, Mat v, Mat f){
    // i is y and j is x
    if(i == -1 || i == rows){
        b[pixC] += v.at<float>(pixC % rows, pixC / rows);
        return 1;
    }
    else if(j == -1 || j == cols){
       b[pixC] += v.at<float>(pixC % rows, pixC / rows);
      return 1;
    }
    else{
        nonZeros.push_back(T(j * rows + i,  pixC, val));
        b[pixC] += f.at<float>(i, j);// - f.at<float>(i, j);
        //b[pixC] += v.at<float>(pixC % rows, pixC / rows);
        return 0;
    }
    
}

Mat poissonReconstruction(Mat img, Mat Gx, Mat Gy){
    assert(Gx.size() == Gy.size());
    Mat v(Gx.rows, Gx.cols, CV_32F);
    Mat f = Mat::zeros(Gx.rows, Gx.cols, CV_32F);
    //calculate second derivative
    Mat dx = Mat::Mat(1, 2, CV_32F);
    dx.at<float>(0,0) = -1; dx.at<float>(0,1) = 1;
    
    Mat dy = Mat::Mat(2, 1, CV_32F);
    dy.at<float>(0,0) = -1; dy.at<float>(1,0) = 1;
    
    Mat Gx2, Gy2;
    filter2D(Gx, Gx2, Gx.depth(), dx);
    filter2D(Gy, Gy2, Gy.depth(), dy);
    
    //calculate laplacian and grad field
    for(int i = 0; i < Gx.rows; i++) for(int j = 0; j < Gx.cols; j++){
        f.at<float>(i, j) = Gx2.at<float>(i, j) + Gy2.at<float>(i, j);
        v.at<float>(i, j) = Gx.at<float>(i, j) + Gy.at<float>(i, j);

    }
    
    //Gx and Gy must be the same size
    //make sparse matrix
    int pixels = Gx.rows * Gx.cols;
    SparseMatrix<float> A(pixels, pixels);
    Eigen::VectorXf b(pixels);
    vector<T> nonZeros;
    int Np, pixC;
    
    for(int i = 0; i < Gx.rows; i++) for(int j = 0; j < Gx.cols; j++){
        pixC = j * Gx.rows + i;//  pixel location in A sparse nxn matrix
        Np = -4;
        Np += insertCoefficients(i, j - 1, pixC, 1, Gx.rows, Gx.cols, nonZeros, b, v, f);
        Np += insertCoefficients(i, j + 1, pixC, 1, Gx.rows, Gx.cols, nonZeros, b, v, f);
        Np += insertCoefficients(i - 1, j, pixC, 1, Gx.rows, Gx.cols, nonZeros, b, v, f);
        Np += insertCoefficients(i + 1, j, pixC, 1, Gx.rows, Gx.cols, nonZeros, b, v, f);
        Np = -4;
        nonZeros.push_back(T(pixC, pixC, Np));
        //b[pixC] += f.at<float>(i, j);
    }
    A.setFromTriplets(nonZeros.begin(), nonZeros.end());
    Eigen::SimplicialCholesky<SparseMatrix<float>> chol(A);
    Eigen::VectorXf x = chol.solve(b);
    
    //recreate image
    Mat I = Mat(Gx.rows, Gy.cols, CV_32F);
    int i, j;
    for(int n = 0; n < pixels; n++){
        i = n % Gx.rows;
        j = (n - i) / Gx.rows;
        I.at<float>(i, j) = x[n];//abs(x[n]);
    }
    
    double minVal, maxVal;
    minMaxLoc(I, &minVal, &maxVal);
    Mat draw;
    I.convertTo(draw, CV_8U, 255/(maxVal - minVal), -minVal * 255 / (maxVal - minVal));
    draw.convertTo(I, CV_32F);
    I /= 255;
    histogramMatching(I, img, I);
    return I;
}