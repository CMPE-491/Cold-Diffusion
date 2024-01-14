# Adversarial Robustness through Cold Diffusion

In this project, we investigate the potential of providing adversarial robustness through a generative model employing various degradation methods. Generative models, rooted in the concept of random noise removal, have shown promising results with Cold Diffusion, where image generation can be achieved without relying on randomness.

Utilizing deterministic forwarding methods, we introduce a specific type of noise to the images. This means that the exact nature of changes in each step is determined, contrasting with random noise. Cold Diffusion presents several approaches, such as blur, animorph, mask, pixelate, and snow, each employing distinct methods and calculations. For this project, we selected the official implementation of the snowification transform from ImageNet-C.

The choice of snowification is motivated by the preservation of pixel information throughout the forward process. Consequently, during the reverse process, we obtain an image that closely resembles the original, effectively rectifying lost pixel information step by step.

Inspired by observations in DENSEPURE and DiffDefense, where attempts to restore noise-added images also aim to correct the effects of adversarial attacks, we seek to enhance the resilience of image processing systems against such attacks. By systematically removing noise, we aim to improve the reliability and integrity of these systems in various applications.

Our methodology involves passing an image through a trained ResNet classifier before the attack. Subsequently, we launch an adversarial attack, ensuring the image is misclassified. Finally, we correct the attack using the Cold Diffusion method, verifying that the image is classified correctly once again.

This project is based on the [Cold Diffusion Models repository](https://github.com/arpitbansal297/Cold-Diffusion-Models/tree/anonymize). We have adapted and extended the provided code to explore the integration of generative models and adversarial robustness in image processing.

## References
* [Cold Diffusion Models Repository](https://github.com/arpitbansal297/Cold-Diffusion-Models/tree/main) by Arpit Bansal