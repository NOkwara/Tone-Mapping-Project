//
//  main.cpp
//  GeometryAnimationFinal
//
//  Created by Nnenna EC Okwara on 10/22/12.
//  Copyright (c) 2012 Nnenna EC Okwara. All rights reserved.
//

#include "ImageProcessing.h"
#include "TonalDistribution.h"
#include "DetailandTexture.h"
#include <set>
//
//  File.cpp
//  interv
//
//  Created by Nnenna EC Okwara on 1/18/13.
//  Copyright (c) 2013 Nnenna EC Okwara. All rights reserved.
//
using namespace std;


/*
 IN EIGEN MEMORY.h Changed commented out line for C++11 support
 */

int main(int argc, char *argv[])
{
    Image image (imread("/Users/NOkwara1/Desktop/url.jpeg", CV_8UC1));
    Image model (imread("/Users/NOkwara1/Desktop/images-5.jpeg", CV_8UC1));
    Mat B, D;
    imshow("start", image.mat);
    imshow("model", model.mat);
    performTonalDistribution(image, model, B, D);
    Mat out = textureness(image, model, B, D);
    detailPres(out, image, model);
    waitKey();
    
    
    return 0;
}

