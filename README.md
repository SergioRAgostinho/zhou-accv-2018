## Introduction

A Python 3 implementation of the absolute pose estimation method for minimal configurations of mixed points and lines introduced by Zhou et al. in

> Lipu Zhou, Jiamin Ye, and Michael Kaess. A Stable Algebraic Camera Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences. In Asian Conference on Computer Vision, 2018.

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

## Interesting Facts

Despite being a Python implementation, the majority of code is written in C. It is a really small section if compared with the extent of the full implementation, but it is way bigger than everything else in its sheer byte size! There are some really long expressions in a very small fraction of the code which Python's interpreter simply couldn't parse, so I've relegated that job to a C compiler.
