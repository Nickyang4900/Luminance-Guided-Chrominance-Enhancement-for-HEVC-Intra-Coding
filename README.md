# Luminance-Guided Chrominance Enhancement for HEVC Intra Coding
The official code implementation of paper 'Luminance-Guided Chrominance Enhancement for HEVC Intra Coding' 

## Abstract
In this paper, we propose a luminance-guided chrominance image enhancement convolutional neural network for HEVC intra coding. Specifically, we firstly develop a gated recursive asymmetric-convolution block to restore each degraded chrominance image, which generates an intermediate output. Then, guided by the luminance image, the quality of this intermediate output is further improved, which finally produces the high-quality chrominance image. When our proposed method is adopted in the compression of color images with HEVC intra coding, it achieves 28.96\% and 16.74\% BD-rate gains over HEVC for the U and V images, respectively, which accordingly demonstrate its superiority

## Training settings for reference
Total epoch = 40, milestone = 20, initial lr = 1e-4, decay = 0.1. 
Training dataset: the first 800 color images of Flickr2K dataset, cropped into 64x64 patches for luminance and 32x32 pathces for chrominance.
Testing dataset: (1)Classical 9-image dataset, (2)McMater 18-image dataset, and (3)Kodak 24-image dataset.

## Experiment Results
BD-rate at 4 QPs = [22,27,32,37]
| Dataset name | w/o Y guidance| with Y guidance  |
| ----------- | ----------- | ----------- |
| Classical      | -9.60%       | -21.92%       |
| McMater   | -10.23%      | -30.99%       |
| Kodak      | -15.23%     | -33.98%       |
| Average   | -11.68%       | -28.96%        |

ΔPSNR at QP 37 
| Dataset name | w/o Y guidance| with Y guidance  |
| ----------- | ----------- | ----------- |
| Classical      | 0.404       | 0.947       |
| McMater   | 0.522    | 1.628      |
| Kodak      | 0.598   | 1.453     |
| Average   | 0.508      | 1.343       |

More details can be seen in paper.
