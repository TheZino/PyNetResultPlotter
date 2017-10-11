# PyNetResultPlotter
Python tool for plotting logs from CNN trainings

- VGG(PERCEPTUAL) and MSE Losses
- PSNR Value
- SSIM Value

For Adversarial plot
- D2x and D4x Losses

## Input Data Format

### For Plot.py
CSV log file: 
* index ; perceptual_loss ; mse_loss ; psnr_value ; ssim_value

### For Plot_adv.py
CSV log file: 
* index ; perceptual_loss ; mse_loss ; Discriminator4x_loss ; Dicriminator2x_loss ; psnr_value ; ssim_value
