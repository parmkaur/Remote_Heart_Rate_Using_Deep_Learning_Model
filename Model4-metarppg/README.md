# Meta-rPPG: Remote Heart Rate Estimation Using a Transductive Meta-Learner

Requirements:

Required Python Packages:
tensorboardX
easydict
tqdm
bypy

The code was tested with python3.6 the following software versions:

| Software     | version | 
| -------------|---------| 
| cuDNN        | 7.6.5   |
| Pytorch      | 1.5.0   |
| CUDA         | 10.2    |

Note: Run Code on a single NVIDIA GTX1080Ti GPU.

Usage:

## Training the model on example.path file containing facial videos of the authors:

To train the model, run train.py file

I ran the training on example.pth file that they linked in the github repository. 

I tried to test with our dataset but it was saying that the maskset should be value between 0-255. But what we are having are mask images.
They did not mention about the maskset input. 

You can find the training results in Result folder.

## Contributing

```
@inproceedings{lee2020meta,
  title={Meta-rPPG: Remote Heart Rate Estimation Using a Transductive Meta-Learner},
  author={Lee, Eugene and Chen, Evan and Lee, Chen-Yi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-MIT-blue)](https://github.com/eugenelet/NeuralScale-Private/blob/master/LICENSE)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)