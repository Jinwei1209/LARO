import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.loss import *


def get_optimizers(netG, netD, lrG, lrD):

    optimizerD = optim.Adam(netD.parameters(), lr = lrD, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = lrG, betas=(0.5, 0.999))

    return optimizerD, optimizerG


def netG_train(
    inputs, 
    targets, 
    AtA,
    netD, 
    netG, 
    optimizerG, 
    lambda_l1=1000, 
    lambda_dc=1000
):

    optimizerG.zero_grad()
    outputs_G = netG(inputs)
    output_D_fake = netD(inputs, outputs_G)
    output_AtA_G = AtA(outputs_G)

    one = Variable(torch.ones(*output_D_fake.size()).cuda())

    loss = loss_classificaiton()
    lossl1 = lossL1()
    lossl2 = lossL2()

    errG_fake = loss(output_D_fake, one)
    errG_l1 = lossl1(outputs_G, targets)
    errG_dc = lossl2(output_AtA_G, inputs)
    errG = errG_fake + lambda_l1 * errG_l1 + lambda_dc*errG_dc

    errG.backward()
    optimizerG.step()

    return  errG_fake.item(), errG_l1.item(), errG_dc.item()


def netD_train(inputs, targets, netD, netG, optimizerD):

    loss = loss_classificaiton()

    optimizerD.zero_grad()
    output_D_real = netD(inputs, targets)
    one = Variable(torch.ones(*output_D_real.size()).cuda())
    errD_real = loss(output_D_real, one)
    # errD_real.backward()

    outputs_G = netG(inputs)
    output_D_fake = netD(inputs, outputs_G)
    zero = Variable(torch.zeros(*output_D_real.size()).cuda())
    errD_fake = loss(output_D_fake, zero)
    # errD_fake.backward()

    errD = errD_fake + errD_real
    errD.backward()

    optimizerD.step()

    return errD.item()/2
