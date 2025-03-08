# Adversarial Robustness through Cold Diffusion

Neural networks are highly susceptible to adversarial attacks, significantly reducing their classification accuracy. This project investigates a generative model known as Cold Diffusion, utilizing deterministic degradation methods, to counteract these attacks and enhance model robustness.

## Goal
Initially, our goal is to apply a deterministic cold diffusion technique (snowification) in the forward process, covering images with simulated snow. A UNet model is trained to revert this effect, thereby reducing the impact of adversarial attacks indirectly. Subsequently, we aim to replace snowification with an actual adversarial attack algorithm, training the UNet to directly reverse these attacks, thus further increasing the classification accuracy.
<table>
  <tr>
    <th align="center">Successful Adversarial Defense via Snowification – Scenario</th>
    <th align="center">Successful Adversarial Defense via FGSM Reversal – Scenario</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/508bc0e7-75c2-46a8-bb4c-a2af984e89a1" width="100%">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/99448990-829e-4927-bcee-01b70c42c647" width="100%">
    </td>
  </tr>
</table>

## Methodology
Two approaches are implemented:

1. Snowification-based degradation: Deterministically apply snowification to degrade images, then use a UNet to restore these images.
2. Adversarial attack-based degradation: Gradually apply FGSM adversarial attacks with increasing severity, training the UNet to directly restore these adversarially degraded images.

## Results
### Snowification-based Approach
In our evaluation, we tested a model trained on 50,000 CIFAR-10 samples using a snowification-based approach to mitigate adversarial attacks. The experiment was conducted over 100,000 epochs with a forward process consisting of 50 timesteps.

During testing, images were altered through an adversarial attack and further modified by adding a snow effect. The UNet model was then used to remove the snow effect, attempting to restore the image and improve classification accuracy.

For each test image, we generated four variations:

1. Original Image – Unaltered input image.
2. Attacked Image – Image with an adversarial attack applied.
3. Snowified Image – Image with added snow applied on top of the adversarial attack.
4. Cleaned Image – Image after the snow has been removed using the UNet model.

![Picture5](https://github.com/user-attachments/assets/f8f2a893-ce45-412b-ae6b-95a6e6a0d651)

### FGSM-based Approach
In this approach, we evaluated the robustness of our model against adversarial attacks generated using the Fast Gradient Sign Method (FGSM). The model was trained on 50,000 CIFAR-10 images over 67,000 epochs and tested on 10,000 images using an FGSM attack with varying epsilon values.

During testing, the FGSM attack was applied gradually over 50 timesteps, starting with a minimum epsilon of 0.01 (2.55/255) and increasing step-by-step to a maximum epsilon of 0.3 (76.5/255). This approach simulates a progressive intensification of the adversarial perturbation, simiilar to snowification of Cold Diffusion. After reaching the maximum attack strength, the UNet model was applied in reverse, using an improved sampling approach to gradually remove the adversarial noise and restore the image.

For each test image, we generated three variations:

1. Original Image – Unaltered input image.
2. Attacked Image – Image with the maximum adversarial attack applied (ε = 0.3).
3. Cleaned Image – Image after the attack has been reversed using the UNet model.

![fgsm_result1](https://github.com/user-attachments/assets/798999c6-616f-4ddc-90e5-5933b6ff7fba)

## Conclusion
Our study demonstrates the effectiveness of Cold Diffusion and UNet-based restoration in mitigating adversarial attacks. By leveraging controlled degradation techniques such as snowification and FGSM-based perturbations, we showed that the UNet model could successfully recover classification accuracy. Notably, the FGSM-based approach achieved a 91.71% recovery, closely matching the baseline accuracy.

These results highlight the potential of generative models in adversarial defense, offering a novel direction for improving neural network robustness. Future work can explore refining these methods further, optimizing training strategies, and extending them to more complex adversarial scenarios.

## References
* [Cold Diffusion Models Repository](https://github.com/arpitbansal297/Cold-Diffusion-Models/tree/main) by Arpit Bansal (We have adapted and extended the provided code to explore the integration of generative models and adversarial robustness in image processing.)
