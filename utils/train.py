import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.loss import *


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


def netD_train(inputs, targets, csms, masks, netD, netG, optimizerD, dc_layer=True):

    loss = loss_classificaiton()

    optimizerD.zero_grad()
    output_D_real = netD(inputs, targets)
    one = Variable(torch.ones(*output_D_real.size()).cuda())
    errD_real = loss(output_D_real, one)
    # errD_real.backward()
    if dc_layer:
        outputs_G = netG(inputs, csms, masks)
    else:
        outputs_G = netG(inputs)
    output_D_fake = netD(inputs, outputs_G)
    zero = Variable(torch.zeros(*output_D_real.size()).cuda())
    errD_fake = loss(output_D_fake, zero)
    # errD_fake.backward()

    errD = errD_fake + errD_real
    errD.backward()

    optimizerD.step()

    return errD_real.item(), errD_fake.item()


def Unet_train(
    inputs, 
    targets, 
    AtA,
    netG, 
    optimizerG, 
    lambda_l1=1000, 
    lambda_dc=1000
):

    optimizerG.zero_grad()
    outputs_G = netG(inputs)
    output_AtA_G = AtA(outputs_G)

    lossl1 = lossL1()
    lossl2 = lossL2()

    errG_l1 = lossl1(outputs_G, targets)
    errG_dc = lossl2(output_AtA_G, inputs)
    errG = lambda_l1 * errG_l1 + lambda_dc*errG_dc

    errG.backward()
    optimizerG.step()

    return errG_l1.item(), errG_dc.item()


def netG_dc_train(
    inputs, 
    targets,
    csms,
    masks,
    netD, 
    netG_dc, 
    optimizerG_dc, 
    lambda_l1=1000
):

    optimizerG_dc.zero_grad()
    outputs_G_dc = netG_dc(inputs, csms, masks)
    output_D_fake = netD(inputs, outputs_G_dc)

    one = Variable(torch.ones(*output_D_fake.size()).cuda())

    loss = loss_classificaiton()
    lossl1 = lossL1()

    errG_dc_fake = loss(output_D_fake, one)
    errG_dc_l1 = lossl1(outputs_G_dc, targets)
    errG_dc = errG_dc_fake + lambda_l1 * errG_dc_l1

    errG_dc.backward()
    optimizerG_dc.step()

    return  errG_dc_fake.item(), errG_dc_l1.item()


def netG_dc_train_no_D(
    inputs,
    targets,
    csms,
    masks,
    netG_dc,
    optimizerG_dc
):

    optimizerG_dc.zero_grad()
    outputs_G_dc = netG_dc(inputs, csms, masks)
    lossl1 = lossL1()
    errG_dc_l1 = lossl1(outputs_G_dc, targets)
    errG_dc_l1.backward()
    optimizerG_dc.step()

    return  errG_dc_l1.item()


def netG_dc_train_intermediate(
    inputs,
    targets,
    csms,
    masks,
    netG_dc,
    optimizerG_dc,
    unc_map
):

    optimizerG_dc.zero_grad()
    if unc_map:
        Xs, Unc_maps = netG_dc(inputs, csms, masks)
    else:
        Xs = netG_dc(inputs, csms, masks)
    lossl2 = lossL2()
    lossl2_sum = 0
    loss_unc_sum = 0

    if unc_map:
        for i in range(len(Xs)):
            temp = (Xs[i] - targets)**2
            lossl2_sum += torch.mean(torch.sum(temp, dim=1)/torch.exp(Unc_maps[i]))
            loss_unc_sum += torch.mean(Unc_maps[i])
        loss_total = lossl2_sum + loss_unc_sum
        loss_total.backward()
        optimizerG_dc.step()
        return lossl2_sum.item(), loss_unc_sum.item()
    else:
        for i in range(len(Xs)):
            lossl2_sum += lossl2(Xs[i], targets)
        lossl2_sum.backward()
        optimizerG_dc.step()
        return  lossl2_sum.item()


def netG_dc_train_pmask(
    opt,
    inputs,
    targets,
    csms,
    brain_masks,
    netG_dc,
    optimizerG_dc,
    unc_map,
    lambda_Pmask
):
    optimizerG_dc.zero_grad()
    if unc_map:
        Xs, Unc_maps = netG_dc(inputs, csms)
    else:
        Xs = netG_dc(inputs, csms)
    lossl2 = lossL2()
    lossl1 = lossL1()
    lossl2_sum = 0
    loss_unc_sum = 0

    if unc_map:
        for i in range(len(Xs)):
            temp = (Xs[i] - targets)**2 * brain_masks
            # lossl2_sum += torch.mean(torch.sum(temp, dim=1)/torch.exp(Unc_maps[i]))
            lossl2_sum += torch.mean(temp/torch.exp(Unc_maps[i]))
            loss_unc_sum += torch.mean(Unc_maps[i])
        loss_Pmask = lambda_Pmask*torch.mean(netG_dc.Pmask)
        loss_total = lossl2_sum + loss_unc_sum + loss_Pmask
        loss_total.backward()
        optimizerG_dc.step()
        return lossl2_sum.item(), loss_unc_sum.item(), loss_Pmask.item()
    else:
        if opt['contrast'] == 'T1':
            for i in range(len(Xs)):
                lossl2_sum += lossl1(Xs[i]*brain_masks, targets*brain_masks)
        elif opt['contrast'] == 'T2':
            for i in range(len(Xs)):
                lossl2_sum += lossl1(Xs[i], targets)
        # lossl2_sum += lossl1(Xs[-1]*brain_masks, targets*brain_masks)
        lossl2_sum.backward()
        optimizerG_dc.step()
        return  lossl2_sum.item(), Xs[-1]