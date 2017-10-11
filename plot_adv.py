import argparse
import csv
import matplotlib.pyplot as plt

# Plot settings
parser = argparse.ArgumentParser(description='Neural Networks data plotter')
parser.add_argument('-f', '--file', type=str, default='./logs/log.csv',
					help='input file to plot (default:./logs/log.csv)')
args = parser.parse_args()

file = args.file
vgg_loss = []
mse_loss = []
psnr = []
ssim = []

adv_D2x=[]
adv_D4x=[]

with open(file) as f:
	reader = csv.reader(f)
	for row in reader:
		#print(row)
		vgg_loss.append(float(row[1]))
		mse_loss.append(float(row[2]))

		adv_D4x.append(float(row[3]))
		adv_D2x.append(float(row[4]))

		psnr.append(float(row[5]))
		ssim.append(float(row[6]))

loss_min_value = min(vgg_loss)

loss_min = [loss_min_value] * len(vgg_loss)

psnr_max_value = max(psnr)

psnr_max = [psnr_max_value] * len(psnr)

ssim_max_value = max(ssim)

ssim_max = [ssim_max_value] * len(ssim)

index_psnr = psnr.index(psnr_max_value)
index_ssim = ssim.index(ssim_max_value)


plt.figure(file + " Adversarial Loss")
a1 = plt.plot(adv_D4x, label="D4x loss")
plt.plot(adv_D2x, label="D2x loss")
plt.legend()
plt.ylabel('loss')
plt.xlabel('epochs')

plt.figure(file + ' Loss and PSNR')
plt.title('Loss')
plt.plot(vgg_loss, label="VGG loss")
plt.plot(mse_loss, label="MSE loss")
plt.plot(loss_min, linestyle = '--', label = str(loss_min_value))
plt.legend()
plt.ylabel('loss')
plt.xlabel('epochs')

plt.figure(file + ' PSNR')
plt.title('PSNR')
plt.plot(psnr, label='AVG PNSR')
plt.plot(psnr_max, linestyle = '--', label = str(psnr_max_value))
plt.plot(index_psnr, psnr_max_value, 'ro')
plt.legend()
plt.ylabel('PSNR')
plt.xlabel('epochs')

plt.figure(file + ' SSIM')
plt.title('SSIM')
plt.plot(ssim, label='AVG SSIM')
plt.plot(ssim_max, linestyle = '--', label = str(ssim_max_value))
plt.plot(index_ssim, ssim_max_value, 'ro')
plt.legend()
plt.ylabel('SSIM')
plt.xlabel('epochs')

plt.show()
