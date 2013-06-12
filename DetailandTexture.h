//
//  DetailandTexture.h
//  GeometryAnimationFinal
//
//  Created by Nnenna EC Okwara on 12/14/12.
//  Copyright (c) 2012 Nnenna EC Okwara. All rights reserved.
//

#ifndef __GeometryAnimationFinal__DetailandTexture__
#define __GeometryAnimationFinal__DetailandTexture__

#include <iostream>
#include "ImageProcessing.h"

Mat textureness(Image I, Image M, Mat &B, Mat &D);
void detailPres(Mat &O, Image I, Image M);
#endif /* defined(__GeometryAnimationFinal__DetailandTexture__) */
