import numpy as np

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    rootName = '/data/Jinwei/T2_slice_recon_GE'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = kdata_loader_GE(
            rootDir=rootName,
            contrast='T2', 
            split='train'
        )
    data_loader = data.DataLoader(data_loader, batch_size=1, shuffle=False)

    
    