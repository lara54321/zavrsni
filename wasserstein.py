import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    # Set random seed for reproducibility
    manual_seed = 999
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ap = argparse.ArgumentParser()
    # ap.add_argument("-o", "--output", required=True,
    #     help="path to output directory")
    ap.add_argument("-e", "--epochs", type=int, default=500,
        help="#epochs to train for")
    ap.add_argument("-b", "--batch-size", type=int, default=128,
        help="batch size for training")
    ap.add_argument("-s", "--image-size", type=int, default=64,
        help="batch size for training")
    ap.add_argument("-lr", "--learning-rate", type=float, default=0.00005,
        help="learning rate for training")
    ap.add_argument("-lower_lr", "--lower-learning-rate", type=int, default=500,
        help="lower learning rate at certain iteration")
    ap.add_argument("-g", "--g_faster", type=int, default=1,
        help="generator learning faster")
    args = vars(ap.parse_args())

    # store the epochs and batch size in convenience variables
    num_epochs = args["epochs"]
    batch_size = args["batch_size"]
    image_size = args["image_size"]
    lr = args["learning_rate"]
    lower_lr = args["lower_learning_rate"]
    g = args["g_faster"]

    # Define the hyperparameters
    dataroot = "images_128"
    workers = 2
    #batch_size = 128
    #image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    #num_epochs = 100
    #lr = 0.00005
    beta1 = 0.5
    ngpu = 1

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Initialize the generator and discriminator
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # Apply the custom weights initialization
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Print the models
    print(netG)
    print(netD)

    # Define the optimizer
    Glr = lr
    Dlr = lr * g
    optimizerD = optim.RMSprop(netD.parameters(), lr=Glr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=Dlr)

    # Training loop
    img_list = []
    G_losses = []
    D_losses = []
    wasserstein_distances = []
    iters = 0
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    n_critic = 1

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize E[D(x)] - E[D(G(z))]
            if iters % lower_lr == 0 and iters != 0:
            # # Lower the learning rate
                Glr = Glr * 0.001
                Dlr = Dlr * 0.001
                optimizerD = optim.RMSprop(netD.parameters(), lr=Glr)
                optimizerG = optim.RMSprop(netG.parameters(), lr=Dlr)
                print('Lowering learning rate.')

            ###########################
            netD.zero_grad()

            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # Forward pass real batch through D
            output_real = netD(real_cpu).view(-1)

            # Generate fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)

            # Forward pass fake batch through D
            output_fake = netD(fake.detach()).view(-1)

            # Wasserstein loss for discriminator
            errD = torch.mean(output_fake) - torch.mean(output_real)
        
            # Store Wasserstein distance
            wasserstein_distance = -errD.item()  # Take the negative value
            wasserstein_distances.append(wasserstein_distance)

            # Update D
            errD.backward()
            optimizerD.step()

            # Clip discriminator weights
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            ############################
            # (2) Update G network: maximize E[D(G(z))]
            ###########################
            if i % n_critic == 0:
                netG.zero_grad()
                output = netD(fake).view(-1)

                # Generator loss
                errG = -torch.mean(output)

                # Update G
                errG.backward()
                optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Wasserstein Distance: {wasserstein_distance:.4f}")

            # Save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        if (epoch % 5 == 0) or (epoch == num_epochs-1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                vutils.save_image(img_list[-1], f"output_wgan/generated_images_{epoch}.png", normalize=True)

        if (epoch % 20 == 0) or (epoch == num_epochs-1):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    vutils.save_image(fake[0], f"output_gp/single/generated_images_{epoch}.png", normalize=True)


    # Save the models
    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')

    # Plot the training losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Generate and save a grid of fake images
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(anim.to_jshtml())
