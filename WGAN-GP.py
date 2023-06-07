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
from torch.autograd import grad

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
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        disc_interpolates = self(interpolates)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


if __name__ == '__main__':
    # Set random seed for reproducibility
    manual_seed = 999
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ap = argparse.ArgumentParser()
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
    lambda_gp = 10

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

  # Create the generator and discriminator
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # Apply weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Set up the loss functions
    #criterion = nn.BCELoss()

    # Set up the optimizer
    #optimizer = optim.Adam(list(netG.parameters()) + list(netD.parameters()), lr=lr, betas=(0.5, 0.999))
        # Print the models
    print(netG)
    print(netD)



    Glr = lr
    Dlr = lr * g
    optimizerD = torch.optim.Adam(netD.parameters(), lr=Glr, betas=(beta1, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=Dlr, betas=(beta1, 0.9))
    # Define the number of updates to discriminator per generator update
    n_critic = 5

    # Create fixed_noise for evaluation during training
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    

    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    wasserstein_distances = []
    gradient_penalties = []

    print("Starting Training Loop...")
    # For each epoch

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            if iters % lower_lr == 0 and iters != 0:
            # # Lower the learning rate
                Glr = Glr * 0.001
                Dlr = Dlr * 0.001
                optimizerD = optim.Adam(netD.parameters(), lr=Glr, betas=(beta1, 0.999))
                optimizerG = optim.Adam(netG.parameters(), lr=Dlr, betas=(beta1, 0.999))
                print('Lowering learning rate.')
            # Update discriminator
            for _ in range(n_critic):
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

                # Compute Wasserstein distance
                wasserstein_distance = torch.mean(output_real) - torch.mean(output_fake)
                wasserstein_distances.append(wasserstein_distance.item())

                # Compute gradient penalty
                gradient_penalty = netD.compute_gradient_penalty(real_cpu, fake.detach())
                gradient_penalties.append(gradient_penalty.item())

                # Update discriminator loss
                errD = wasserstein_distance + lambda_gp * gradient_penalty

                # Backward pass and optimizer step
                errD.backward()
                optimizerD.step()

            # Update generator
            netG.zero_grad()
            output_fake = netD(fake).view(-1)
            errG = -torch.mean(output_fake)

            # Backward pass and optimizer step
            errG.backward()
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        # Print epoch progress
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Generator Loss: {errG.item():.4f}, "
            f"Discriminator Loss: {errD.item():.4f}, "
            f"Wasserstein Distance: {wasserstein_distance.item():.4f}, "
            f"Gradient Penalty: {gradient_penalty.item():.4f}")
        if (epoch % 5 == 0) or (epoch == num_epochs-1):
            with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    vutils.save_image(img_list[-1], f"output_gp/generated_images_{epoch}.png", normalize=True)
        if (epoch % 20 == 0) or (epoch == num_epochs-1):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    vutils.save_image(fake[0], f"output_gp/single/generated_images_{epoch}.png", normalize=True)

        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="Generator")
        plt.plot(D_losses, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Animation showing the generated images over time
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())

        # Save the generator model
        torch.save(netG.state_dict(), "WGAN_GP_generator.pth")