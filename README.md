Tone-Mapping-Project
====================

Using tone-mapping of second image maps it to original.

Algorithm:

Large-Scale Tonal Distribution

Bilateral Decomposition

This decomposes the photograph into two components Base, B, and Detail, D for the image, I.
B = bf(I) 
D = I– B.

bf is the bilateral filter of the image, which is useful as it preserves the details of an image better than general Gaussian smoothing. In doing the bilateral filter it is important to choose a good σs and σr of the image for this implementation they were chosen as min(width, height) / 16 and p90(||∇I||), 90th percentile of the gradient norm, respectively. 

Gradient Reversal Removal
 In Bilateral Decomposition the amount of detail in the image can be increased around certain edges. 
 Therefore it is important to correct these artifacts in the Detail image’s derivatives in both the x
 and y position. After correcting them I use Poisson Reconstruction to rebuild the image without 
 artifacts. I then get a new B with B = I – D.

Tonal Balance
As the base layer contains large-scale spatial distribution of tones, we wont to continue this property 
so that this distribution matched the model image. We do this with histogram matching to transfer the 
histogram of the base image to that of the model image.

Detail and Texture Management

Detail Management Based on Frequency Analysis
Because of the Gaussian filter range in bilateral filters, high-frequency and high-contrast will mostly 
be in the base. This is a problem for detail managementThis problem needs to be corrected by analyzing 
the high frequency in both the detail and base layer to ensure that all frequency content is taken into 
account. This information is used to decide how to update the detail layer.

Textureness

We want to characterize local amounts of high frequency content over the image to distinguish regions 
with high levels of detail from smooth regions. To do this we compute a high-pass version, H, of the 
image using the cutoff σs. and take the magnitude of H. To avoid the halo effect visible in linear 
frequency decomposition of image intensities we use cross-bilateral filters where the intensity image
defines the edge-preserving term to filter |H|. This textureness is defined as:

 T(I)p = (1/k) sum(gσs(||p - q||) gσr(|Ip - Iq|) |H|q)
 
 where k = sum(gσs(||p - q||) gσr(|Ip - Iq|))

we use the same σr and σs 8 times larger to ensure smooth textureness variations on uniform regions

Textureness Transfer
From above we have the textureness maps of both the Model and Image T(m) and T(I), respectively. 
We use histogram matching to build Textureness map T’. To prevent halos a scheme was designed to 
understand the affect the detail layer should have on the output image using the map T’:

ρd = max(0, (T'p - T(B')p) / T(D)p)

By not including negative ratios, we are preventing gradient reversals allowing the linear combination 
of the detail and base layer as O = B + ρD.

Detail Preservation
The output image created above has the opportunity to be saturated in highlights and shadows. To create 
crisp details in the image we enforce the intensity histogram of the model M to the current output, 
returning the values to the displayable range. Secondly, we change the gradient field to ensure that no 
details are removed or overly emphasized. This is similar to what is done in Gradient Reversal Removal.  
Finally we use the Poisson Reconstruction to rebuild the image.

