import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 6, 7"
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PairedDataSet import PairedData
from Model_RAT import RAT
from losses import CharbonnierLoss, FocalRegionLoss, PerceptualLoss, AdversarialLoss
from utils import tensor2img, save_img, save_itk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import contingency_table
from utils import calculate_psnr, calculate_ssim

import warnings
warnings.simplefilter("ignore")
import time
from torch.amp import autocast, GradScaler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def mkdir(p):
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p))
def save_model(model, save_path, suf=''):
    save_path = os.path.join(save_path,'model_{}.pth'.format(suf))
    torch.save(model.cpu().module.state_dict(), save_path)
    model.cuda()



def cal_psnr_numpy(img1, img2, data_range=255):
    B, H, W, C = img1.shape
    mse = (img1 - img2) ** 2
    mse = np.mean(mse.reshape(B, -1), axis=1)
    return list(10 * np.log10(data_range**2/mse))

def save_multiple_img(img, name_list, save_path):
    B = img.shape[0]
    for i in range(B):
        save_img(img[i], os.path.join(save_path, name_list[i]))




def plot_psnr(data, label, save_path ):
    l = len(data)
    axis = np.linspace(1, l, l)

    fig = plt.figure()
    plt.title(label)

    plt.plot(
        axis,

        data,
        label=label
    )
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(label)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close(fig)


set_seeds()


data_root = r"D:\research\su\png"
mask_root = r"D:\research\su\mask"
save_path = r"D:\research\su\results1"
fine_tune = r"D:\research\su\fine_tune"
mkdir(save_path)
mkdir(fine_tune)




batch_size = 2
lr = 2e-4
train_iter = 2e5
eval_interval = 1000
loss_beta = 1e-3

# '''
# Prepare DataLoader
# '''
# print('Load Data')

data_set = {'train':PairedData(root=data_root, mask_root= mask_root,target='train', use_fine_mask = True, use_coarse_mask=False),
            'test': PairedData(root=data_root, mask_root= mask_root, target='test', use_fine_mask = True, use_coarse_mask=False, use_num=100),
            }
data_loader = {
    'train': DataLoader(data_set['train'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data_set['test'], batch_size=batch_size)
}
datasize = {'train': len(data_loader['train'].dataset),
            'test': len(data_loader['test'].dataset)
            }
# print(datasize)


# '''
# Prepare Model
# '''

loss_fun = FocalRegionLoss(beta=loss_beta)

G=RAT(loss_fun = loss_fun)

if torch.cuda.is_available():
    G.cuda()
G = nn.DataParallel(G, device_ids=range(torch.cuda.device_count()))
optimizer_G = torch.optim.Adam(G.parameters(),lr = lr, betas = (0.5, 0.999))
lr_scheduler_G = CosineAnnealingLR(optimizer_G, train_iter, eta_min=1.0e-6)

# '''
# Train Model
# '''

record_metrics={
    "psnr":[],
    "loss":[]
    }


epoch_num = int(np.ceil(train_iter/np.floor(datasize['train']/batch_size)))

pbar = tqdm(total=train_iter)
psnr_max = 0

scaler = GradScaler('cuda')

for epoch in range(epoch_num):
    num = 0
    for data in enumerate(data_loader['train']):
        num+=1
        # print('No.',num)
        # print(len(data))
        hr_img, lr_img, input_mask, _, name_list = data[1]
        # print('Data shape:',data.shape)
        lr_img = lr_img.type(torch.FloatTensor).cuda()
        hr_img = hr_img.type(torch.FloatTensor).cuda()
        input_mask = input_mask.type(torch.FloatTensor).cuda()

        # import pdb
        # pdb.set_trace()

        #################
        #     train G
        #################

        G.train()
        optimizer_G.zero_grad()
        # print(lr_img.shape)
        # print(input_mask.shape)
        # print(hr_img.shape)

        with autocast('cuda'):
            loss_G = G(lr_img, input_mask, hr_img).mean()

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        record_metrics['loss'].append(loss_G.item())
        torch.cuda.empty_cache()
        lr_scheduler_G.step()

        counter = pbar.n

        if counter % eval_interval==0:
            psnr=[]
            G.eval()
            with torch.no_grad():
                for _,data in enumerate(tqdm(data_loader['test'])):
                    # print(data)
                    hr_img, lr_img, input_mask, _, name_list = data
                    lr_img = lr_img.type(torch.FloatTensor).cuda()
                    input_mask = input_mask.type(torch.FloatTensor).cuda()
                    # import pdb
                    # pdb.set_trace()
                    sr_img = G(lr_img, input_mask).detach().cpu()
                    torch.cuda.empty_cache()
                    sr_img = tensor2img(sr_img)
                    hr_img = tensor2img(hr_img)
                    psnr += cal_psnr_numpy(sr_img, hr_img, data_range=255)

                    file_path = os.path.join(save_path, 'val_img')
                    mkdir(file_path)

                    save_multiple_img(sr_img, name_list, file_path)
                    # import pdb
                    # pdb.set_trace()

            psnr_mean = np.mean(psnr)
            record_metrics['psnr'].append(psnr_mean)
            if psnr_mean>psnr_max:
                psnr_max = psnr_mean

                file_path = os.path.join(save_path, 'checkpoint')
                mkdir(file_path)
                save_model(G, file_path, suf='best')

            if counter>train_iter-eval_interval*100:
                save_model(G, file_path, suf=str(counter))

            plot_psnr(record_metrics['loss'], 'train loss', os.path.join(save_path, 'train_loss.pdf'))
            plot_psnr(record_metrics['psnr'], 'val psnr', os.path.join(save_path, 'val_psnr.pdf'))

        pbar.set_description("loss:{:6}, psnr:{:6}".format(record_metrics['loss'][-1], record_metrics['psnr'][-1]))
        pbar.update()

        if counter>=train_iter:
            break

    if counter>=train_iter:
        break

with open(os.path.join(save_path, "record_info.bin"),'wb') as f:
    pickle.dump(record_metrics, f)

'''
fine tune
'''

checkpoint = torch.load(os.path.join(save_path, 'checkpoint', "model_best.pth"), map_location=device)
G.load_state_dict(checkpoint, strict=False)

G.train()


num_epochs = 50
initial_lr = 1e-5
lr_scheduler_step = 10
lr_scheduler_gamma = 0.5 # decay factor

optimizer_G = torch.optim.Adam(G.parameters(), lr=initial_lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)


for epoch in tqdm(num_epochs):
    num = 0
    for counter, data in tqdm(enumerate(data_loader['train'])):
        num += 1
        hr_img, lr_img, input_mask, _, name_list = data
        lr_img = lr_img.type(torch.FloatTensor).cuda()
        hr_img = hr_img.type(torch.FloatTensor).cuda()
        input_mask = input_mask.type(torch.FloatTensor).cuda()

        optimizer_G.zero_grad()

        with autocast('cuda'):
            loss_G = G(lr_img, input_mask, hr_img).mean()

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        record_metrics['loss'].append(loss_G.item())
        torch.cuda.empty_cache()

        lr_scheduler_G.step()

        if num % eval_interval == 0:
            psnr = []
            G.eval()
            with torch.no_grad():
                for _, data in enumerate(tqdm(data_loader['test'])):
                    hr_img, lr_img, input_mask, _, name_list = data
                    lr_img = lr_img.type(torch.FloatTensor).cuda()
                    input_mask = input_mask.type(torch.FloatTensor).cuda()
                    sr_img = G(lr_img, input_mask).detach().cpu()
                    sr_img = tensor2img(sr_img)
                    hr_img = tensor2img(hr_img)
                    psnr += cal_psnr_numpy(sr_img, hr_img, data_range=255)

            psnr_mean = np.mean(psnr)
            record_metrics['psnr'].append(psnr_mean)
            print(f'Epoch [{epoch+1}/{num_epochs}], PSNR: {psnr_mean:.4f}')

with open(os.path.join(fine_tune, "record_info.bin"), 'wb') as f:
    pickle.dump(record_metrics, f)

'''
test image
'''

test_dataset="Thyroid"

checkpoint = torch.load(os.path.join(fine_tune, 'checkpoint', "model_best.pth"),map_location=device)
# print(checkpoint)

G.load_state_dict(checkpoint, strict=False)
G = torch.nn.DataParallel(G)
G.eval()
time_list = []
psnr_values = []
ssim_values = []

with torch.no_grad():
    for counter, data in enumerate(tqdm(data_loader['test'])):
        hr_img, lr_img, mask_fine, mask_coase, name_list = data

        lr_img = lr_img.type(torch.FloatTensor).cuda()
        input_mask = mask_fine.type(torch.FloatTensor).cuda()

        sr_img = G(lr_img, input_mask).detach().cpu()
        sr_img = tensor2img(sr_img)

        print("sr_img shape:", sr_img.shape)

        # compute PSNR and SSIM
        hr_img = tensor2img(hr_img)
        psnr_value = calculate_psnr(sr_img, hr_img, crop_border=0)
        psnr_values.append(np.mean(psnr_value))

        ssim_value = calculate_ssim(sr_img, hr_img)
        ssim_values.append(ssim_value)

        file_path = os.path.join(save_path, "test_img_{}".format(test_dataset))
        mkdir(file_path)
        save_multiple_img(sr_img, name_list, file_path)
        torch.cuda.empty_cache()

# averaged
print("PSNR:",psnr_values)
print("Average PSNR:", np.mean(psnr_values))
print("SSIM:", ssim_values)
print("Average SSIM:", np.mean(ssim_values))