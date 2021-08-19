# Multiscale SSIM (MS-SSIM)

Allows you to train using the MS-SSIM (multiscale structural similarity index measure) as the main loss metric,
a perceptually more accurate measure of image quality than MSE (mean squared error).

As an added benefit, you may see a decrease in ms/iteration (when using the same batch size) with Multiscale loss
enabled. You may also be able to train with a larger batch size with it enabled.

- [DESCRIPTION](#description)
- [USAGE](#usage)

## DESCRIPTION

[SSIM](https://en.wikipedia.org/wiki/Structural_similarity) is metric for comparing the perceptial quality of an image:
> SSIM is a perception-based model that considers image degradation as perceived change in structural information, 
> while also incorporating important perceptual phenomena, including both luminance masking and contrast masking terms. 
> [...]
> Structural information is the idea that the pixels have strong inter-dependencies especially when they are spatially 
> close. These dependencies carry important information about the structure of the objects in the visual scene. 
> Luminance masking is a phenomenon whereby image distortions (in this context) tend to be less visible in bright 
> regions, while contrast masking is a phenomenon whereby distortions become less visible where there is significant 
> activity or "texture" in the image.

The current loss metric is a combination of SSIM (structural similarity index measure) and 
[MSE](https://en.wikipedia.org/wiki/Mean_squared_error) (mean squared error).

[Multiscale SSIM](https://en.wikipedia.org/wiki/Structural_similarity#Multi-Scale_SSIM) is a variant of SSIM that
improves upon SSIM by comparing the similarity at multiple scales (e.g.: full-size, half-size, 1/4 size, etc.)
By using MS-SSIM as our main loss metric, we should expect the image similarity to improve across each scale, improving
both the large scale and small scale detail of the predicted images.

Original paper: [Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. 
"Multiscale structural similarity for image quality assessment." 
Signals, Systems and Computers, 2004.](https://www.cns.nyu.edu/pub/eero/wang03b.pdf)

## USAGE

```
[n] Use multiscale loss? ( y/n ?:help ) : y
```



