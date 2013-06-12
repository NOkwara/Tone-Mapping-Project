//
//  TonalDistribution.cpp
//  GeometryAnimationFinal
//
//  Created by Nnenna EC Okwara on 12/14/12.
//  Copyright (c) 2012 Nnenna EC Okwara. All rights reserved.
//

#include "TonalDistribution.h"

int sign(float x){
    return (x > 0) - (x < 0);
}
void performTonalDistribution(Image I, Image M, Mat &B, Mat &D){
    
   //I.mat in 32F
    B = I.bf(I);
    D = Mat::zeros(B.rows, B.cols, CV_32F);
    for(int i = 0; i < I.mat.rows; i++) for(int j = 0; j < I.mat.cols; j++){
        D.at<float>(i, j) = I.mat.at<float>(i, j) - B.at<float>(i, j);
    }
   
    //create both dx and dy images for D and I
    Mat Ddx, Ddy;
    
    Mat Idx, Idy;
    
    Mat dx = Mat::Mat(1, 2, CV_32F);
    dx.at<float>(0,0) = -1; dx.at<float>(0,1) = 1;
    
    Mat dy = Mat::Mat(2, 1, CV_32F);
    dy.at<float>(0,0) = -1; dy.at<float>(1,0) = 1;
    
    filter2D(D, Ddx, D.depth(), dx);
    filter2D(D, Ddy, D.depth(), dy);
    
    filter2D(I.mat, Idx, I.mat.depth(), dx);
    filter2D(I.mat, Idy, I.mat.depth(), dy);
    
    //update gradient fields using proper requirements
    for(int i=0; i < D.rows; i++) for(int j = 0; j < D.cols; j++){
        //Doing Dx
        if(sign(Ddx.at<float>(i, j)) != sign(Idx.at<float>(i, j)))
            Ddx.at<float>(i, j) = 0;
        else if(abs(Ddx.at<float>(i, j)) > abs(Idx.at<float>(i, j)))
            Ddx.at<float>(i, j) = Idx.at<float>(i, j);
        else{}
        
        //Doing Dy
        if(sign(Ddy.at<float>(i, j)) != sign(Idy.at<float>(i, j)))
            Ddy.at<float>(i, j) = 0;
        else if(abs(Ddy.at<float>(i, j)) > abs(Idy.at<float>(i, j)))
            Ddy.at<float>(i, j) = Idy.at<float>(i, j);
        else{}
    }
    
    //do Poisson Reconstruction to get ret of fixed B and D
    D = poissonReconstruction(D, Ddx,Ddy);
    //Get it back to logarithm values 0 to 1
    double minVal, maxVal;
    minMaxLoc(D, &minVal, &maxVal);
    Mat draw;
    D.convertTo(draw, CV_8U, 255/(maxVal - minVal), -minVal * 255 / (maxVal - minVal));
    draw.convertTo(D, CV_32F);
    D /= 255;
    for(int i = 0; i < I.mat.rows; i++) for(int j = 0; j < I.mat.cols; j++){
        B.at<float>(i, j) = abs(I.mat.at<float>(i, j) - D.at<float>(i, j));
    }
   
    //Tonal Balance
    Mat Bm = M.bf(M);
    histogramMatching(B.clone(), M.mat, B);
}