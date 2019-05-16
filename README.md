## Introduction

A Python 3 implementation of the absolute pose estimation method for minimal configurations of mixed points and lines introduced by Zhou et al. in

> Lipu Zhou, Jiamin Ye, and Michael Kaess. A Stable Algebraic Camera Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences. In Asian Conference on Computer Vision, 2018.

The methods implemented are able to address 4 modalities of minimal problems, namely: perspective-3-points (P3P), perspective-2-points-1-line (P2P1L), perspective-1-point-2-lines (P1P2L) and perspective-3-lines (P3L).
This package also includes an implementation the robustified version of Kukelova et al. E3Q3 method presented in


 > Zuzana Kukelova, Jan Heller, and Andrew Fitzgibbon. Efficient intersection of three quadrics and applications in computer vision. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.


They are clearly isolated, so if you're only interested in E3Q3 you can import it separate from everything else.
```python
from zhou_accv_2018 import e3q3
```

**License:** Apache 2.0

## Installing

Clone this repo and invoke from its root folder
```
python setup.py install
```

## Examples

The library exposes 5 public functions: `p3p`, `p3l`, `p2p1l`, `p1p2l`, and `e3q3`. You can find a couple of examples showing how to use each in the [examples folder](https://github.com/SergioRAgostinho/zhou-accv-2018/blob/master/examples).

## Interesting Facts

Despite being a Python implementation, the majority of code is written in C. It is a really small section if compared with the extent of the full implementation, but it is way bigger than everything else in its sheer byte size! There are some really long expressions in a very small fraction of the code which Python's interpreter simply couldn't parse, so I've relegated that job to a C compiler.
