# Human posture estimation project

### citations:
```
 @software{Gorordo_Fernandez_pyKinectAzure,
    author = {Gorordo Fernandez, Ibai},
    title = {{pyKinectAzure}}
    }
 @InProceedings{Huang_2020_CVPR,
    author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
    title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }
 @article{huang2020aid,
    title={AID: Pushing the Performance Boundary of Human Pose Estimation with Information Dropping Augmentation,
    author={Huang, Junjie and Zhu, Zheng and Huang, Guan and Du, Dalong},
    journal={arXiv preprint arXiv:2008.07139},
    year={2020}
    }
```
### instructions:
1. This task use multiple azure kinects(abbreviated AK) to estimate human posture with its own SDK.
2. AKs’ estimations are as observation for Belief Propagation(abbreviated BP).
3. Using UDP-Pose's estimations to get more observation for BP.
4. Realtime(in realtime.py) and offline(in dataProcess.py [def BPapplication]) mode are available.
5. Not all defs are used, some defs was used for testing datas or something else.
6. To be replenished.