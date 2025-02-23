# Region Attention Transformer for Medical Image Restoration (RAT)

PyTorch implementation for Region Attention Transformer for Medical Image Restoration [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.09268) (MICCAI 2024).

## Network Architecture

![](README.assets/architecture.JPG)

## Visual Comparison

![](README.assets/vis.JPG)

## Getting Started with Model Inference

RAT has two inputs: the **input image** and the **indexed mask** obtained from post-processing the SAM output mask. The example input image is located at “./example_img/input_img.png", and the resulting indexed mask can be found at “./example_img/indexed_mask.nii”.

Next, we will first explain how to obtain the indexed mask using SAM, followed by an introduction to the final model inference.

- **Mask Prediction & Postprocess**

  First, you can obtain region partitioning masks with the Segment Anything Model ([SAM](https://github.com/facebookresearch/segment-anything)) as follows:

  ```python
  from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
  sam = sam_model_registry["<model_type>"](checkpoint = "<path/to/checkpoint>")
  mask_generator = SamAutomaticMaskGenerator(sam)
  masks = mask_generator.generate(<your_image>) 
  #<your_image>: Load the input image at "./example_img/input_img.png"
  ```

  !!!!! It is now recommended to employ advanced models such as  [Efficient-SAM](https://github.com/yformer/EfficientSAM), and [RWKV-SAM](https://github.com/HarborYuan/ovsam), as they provide superior efficiency and effectiveness over the conventional SAM model.

  Then, you need to post-process the masks to obtain an indexed mask, which can be then used for compact region partitioning during the downsampling process.

  ```python
  import numpy as np
  def toSegMap(masks): 
      result = np.zeros(masks[0]['segmentation'].shape)
      for i in range(len(masks)): 
          result[masks[i]['segmentation']] = (i+1) 
      result[result==0] = len(masks) + 1
      return result
  masks = sorted(masks, key = itemgetter('area'), reverse = True) 
  indexed_mask = toSegMap(masks) # Output at "./example_img/indexed_mask.nii"
  ```

  The resultant indexed mask is available at "./example_img/indexed_mask.nii". You can use [AMIDE](https://amide.sourceforge.net/) or [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) softwares to visualize the ".nii" file. To facilitate understanding, a toy example of the indexed mask is displayed below:

  ![](README.assets/toy_indexed_mask.png)

- **RAT Inference**

  With the input image and its resultant indexed mask, the output of  RAT can be obtained as follows:

  ```python
  from Model_RAT import RAT
  model = RAT()
  output_img = model(input_img, indexed_mask) 
  # lr_img shape: [B, C, H, W] 
  # indexed_mask shape: [B, H, W]
  ```

## Citation

If you find RAT useful in your research, please consider citing:

```bibtex
@inproceedings{yang2024rat,
  title={Region Attention Transformer for Medical Image Restoration},
  author={Yang, Zhiwen and Chen, Haowei and Qian, Ziniu and Zhou, Yang and Zhang, Hui and Zhao, Dan and Wei, Bingzheng and Xu, Yan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={603--613},
  year={2024},
  organization={Springer}
}
```

