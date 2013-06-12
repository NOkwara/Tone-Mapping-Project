//
//  DetailandTexture.cpp
//  GeometryAnimationFinal
//
//  Created by Nnenna EC Okwara on 12/14/12.
//  Copyright (c) 2012 Nnenna EC Okwara. All rights reserved.
//

#include "DetailandTexture.h"
const int channels = 256;

float calculatephi(Mat O, Mat I){
    int Ihist[channels] = {0};
    int Ohist[channels] = {0};
    
    double minVal, maxVal;
    minMaxLoc(I, &minVal, &maxVal);
    //double dist = maxVal - minVal; //values between max val and min val
    float fraction = (maxVal - minVal) / 255.0; // 255 is number of channel to split*/
    
	//create historgram fo red blue and green channels
	for(int x = 0; x < I.rows; x++) for(int y = 0; y < I.cols; y++){
        if(round(I.at<float>(x, y) / fraction) > 255)
            Ihist[255]++;
        else if(round(I.at<float>(x, y) / fraction) < 0)
            Ihist[0]++;
        else{
            Ihist[(int) round(I.at<float>(x, y) / fraction)]++;
        }
	}
    
    minMaxLoc(I, &minVal, &maxVal);
    
    fraction = (maxVal-minVal) / 255.0;
    
    //create historgram fo red blue and green channels
	for(int x = 0; x < O.rows; x++) for(int y = 0; y < O.cols; y++){
        if(round(O.at<float>(x, y) / fraction) > 255)
            Ohist[255]++;
        else if(round(O.at<float>(x, y) / fraction) < 0)
            Ohist[0]++;
        else{
            Ohist[(int) round(O.at<float>(x, y) / fraction)]++;
        }
	}
    //normalize both
    
    float totalO = 0;
    float totalI = 0;
    
    for(int i = 0; i < 256; i++){
        totalI += Ihist[i];
        totalO += Ohist[i];
    }
    
    for(int i = 0; i < channels; i++){
        Ihist[i] /= totalI;
        Ohist[i] /= totalO;
    }
    totalI = 0;
    totalO = 0;
    float I05, I95, O05, O95;
    bool done5 = false;
    //find the .05 and .95 mark for each
    for(int i = 0; i < channels; i++){
        totalI += Ihist[i];
        if(totalI > 0.05 && !done5){
            done5 = true;
            I05 = i;
        }
        if(totalI > 0.95){
            I95 = i;
            break;
        }
    }
    done5 = false;
    
    for(int i = 0; i < channels; i++){
        totalO += Ohist[i];
        if(totalO > 0.05 && !done5){
            done5 = true;
            O05 = i;
        }
        if(totalO > 0.95){
            O95 = i;
            break;
        }
    }
    
    return (O95 - O05) / (I95 - I05);
}
Mat textureness(Image I, Image M, Mat &B, Mat &D){
    Mat H = I.highPassFilter();

    //get back to log values
    
    Mat Hm = M.highPassFilter();
    
    I.setSigmaS(8 * I.getSigmaS());
    Mat TofI = I.bf(Image(H, 1));
    
    
    double minVal, maxVal;
     minMaxLoc(TofI, &minVal, &maxVal);
     Mat draw;
     TofI.convertTo(draw, CV_8U, 255/(maxVal - minVal), -minVal * 255 / (maxVal - minVal));
     draw.convertTo(TofI, CV_32F);
     TofI /= 255;
    
    M.setSigmaS(8 * I.getSigmaS());
    Mat TofM = M.bf(Image(Hm, 1));
    
    minMaxLoc(TofM, &minVal, &maxVal);
    TofM.convertTo(draw, CV_8U, 255/(maxVal - minVal), -minVal * 255 / (maxVal - minVal));
    draw.convertTo(TofM, CV_32F);
    TofM /= 255;
    
    Mat Tprime = Mat(I.mat.rows, I.mat.cols, CV_32F);
    histogramMatching(TofI, TofM, Tprime); //perform historgram transfer
    
    Image base (B, 1);
    base.setSigmaS(base.getSigmaS() * 8);
    
    Image detail (D, 1);
    detail.setSigmaS(detail.getSigmaS() * 8);
    
    Image htext (base.highPassFilter(), 1);
    Image hmText (detail.highPassFilter(), 1);
    
    
    Mat TofB = base.bf(htext);
    Mat TofD = detail.bf(hmText);
    
    Mat output = Mat::zeros(I.mat.rows, I.mat.cols, CV_32F);
    float rho = 0;
    for(int i = 0; i < I.mat.rows; i++) for(int j = 0; j < I.mat.cols; j++){
        if(TofD.at<float>(i,j) == 0)
            rho = 0;
        else{
            rho = (Tprime.at<float>(i, j) - TofB.at<float>(i, j))/ TofD.at<float>(i,j);
            rho = 0 > rho ? 0 : rho;
        }
        output.at<float>(i, j) = B.at<float>(i, j) + rho * D.at<float>(i, j);
    }
    return output;
    
    
}

void detailPres(Mat &O, Image I, Image M){
    //first do intensity transfer with Model image
    histogramMatching(O.clone(), M.mat, O);
    
    imshow("histogram", O);
    
    Mat Odx, Ody;
    
    Mat Idx, Idy;
    
    Mat dx = Mat::Mat(1, 2, CV_32F);
    dx.at<float>(0,0) = -1; dx.at<float>(0,1) = 1;
    
    Mat dy = Mat::Mat(2, 1, CV_32F);
    dy.at<float>(0,0) = -1; dy.at<float>(1,0) = 1;

    filter2D(O, Odx, O.depth(), dx);
    filter2D(O, Ody, O.depth(), dy);
    
    filter2D(I.mat, Idx, I.mat.depth(), dx);
    filter2D(I.mat, Idy, I.mat.depth(), dy);
    
    float phi = calculatephi(O, I.mat);
    float alpha = phi / 4.0, beta = 1 + 3*vTau(phi);
    //for gradient fixing for both x and y gradient
    for(int i = 0; i < O.rows; i ++) for(int j = 0; j < O.cols; j++){
        if(abs(Odx.at<float>(i, j)) < alpha * abs(Idx.at<float>(i, j)))
            Odx.at<float>(i ,j) = alpha * Idx.at<float>(i, j);
        else if(abs(Odx.at<float>(i, j)) > beta * abs(Idx.at<float>(i, j)))
            Odx.at<float>(i ,j) = alpha * Idx.at<float>(i, j);
        else{}
    
        if(abs(Ody.at<float>(i, j)) < alpha * abs(Idy.at<float>(i, j)))
            Ody.at<float>(i ,j) = alpha * Idy.at<float>(i, j);
        else if(abs(Ody.at<float>(i, j)) > beta * abs(Idy.at<float>(i, j)))
            Ody.at<float>(i ,j) = alpha * Idy.at<float>(i, j);
        else{}
    }
    //rebuild using Poisson Reduction
    O = poissonReconstruction(O, Odx, Ody);
    imshow("end", O);
    double minVal, maxVal;
    minMaxLoc(O, &minVal, &maxVal);
    Mat draw;
    O.convertTo(draw, CV_8U, 255/(maxVal - minVal), -minVal * 255 / (maxVal - minVal));
    
    //waitKey();
}