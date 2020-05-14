# LOUPE-ST

For Bayesian Optimization project, use:

```
python main_EI.py 
```
to check the result. You need to install 'gpytorch' to run the code. 

Please run the code in GPU server, since the data are located there.


Recon solver:

I implemented the iterative reconstruction solvers for under-sampled k-space recon. In 
```
/bayesOpt/sample_loss/recon_loss function
```
the command
```
model = DC_ST_Pmask(input_channels=2, filter_channels=32, lambda_dll2=1e-4, 
                        lambda_tv=1e-4, rho_penalty=1e-2, flag_ND=3, flag_solver=-3, 
                        flag_TV=1, K=20, rescale=True, samplingRatio=sampling_ratio, flag_fix=1, pmask_BO=p_pattern)
```
build the solver with a bunch of parameters, where
```
K=20
```
means the quasi-newton outer loop iteration numbers are 20. You can change this parameter to introduce 'reconstruction noise' as we discussed. 


Kspace dataloader:
```
loader/kdata_loader_GE/kdata_loader_GE
```
is the dataloader function for under-sampled kspace data. You can add noise in kspace in the lines:
```
kdata = np.transpose(kdata, (2, 0, 1))
kdata = c2r_kdata(kdata)
```

New implementations:
If you have new modules implemented, put them in 
```
/bayesOpt
```
folder.

