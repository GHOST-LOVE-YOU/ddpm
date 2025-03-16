## quick glance

### version1

trained a simplest Unet network without ddpm

input image => add random noise => predict original image

![version1_prediction](./image/version1_model_predictions.png)

### version2

Unet + ddpm

![version2_prediction](./image/version2_model_predictions.png)

### version3

Unet + ddpm + ddim

![version3_prediction](./image/version3_model_predictions.png)

## thanks

- [8bit-diffusion-model](https://github.com/brain-xiang/8bit-diffusion-model/tree/main)
- [huggingface|diffusion course](https://huggingface.co/learn/diffusion-course/unit0/1)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) , [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
