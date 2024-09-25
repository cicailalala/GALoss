import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
from utils import metrics, ramps, test_amos_vnet_AB, cube_losses, cube_utils
from dataloaders.dataset import *
from networks.magicnet import VNet_Magic
from networks.vnet import VNet
from torch import nn
import os
from GALoss import GADice, GACE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='AMOS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./data/AMOS/', help='Name of Dataset')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save')
parser.add_argument('--exp', type=str, default='CPS', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=17000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=360, help='total samples of the dataset')
parser.add_argument('--max_train_samples', type=int, default=340, help='maximum samples to train')
parser.add_argument('--max_test_samples', type=int, default=120, help='maximum samples to test')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=10, help='labeled trained samples')
# 4 for 2%, 10 for 5%
parser.add_argument('--seed', type=int, default=1337, help='random seed') # 1337, 0, 666
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--block_size', type=int, default=32, help='size of each block for calculate dice')
parser.add_argument('--lamda', type=float, default=0.2, help='weight to balance all losses')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
args = parser.parse_args()


def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, max_epoch):
    return 0.1 * sigmoid_rampup(epoch, max_epoch)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)


def create_model(n_classes=16, ema=False):
    # Network definition
    net =  VNet(n_classes=num_classes)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model




def read_list(split):
    ids_list = np.loadtxt(
        os.path.join('./data/amos_splits/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)

if args.labelnum == 4:
    # 2%, 4 labeld 
    labeled_list = read_list('labeled_2p')
    unlabeled_list = read_list('unlabeled_2p')
elif args.labelnum == 10:
    # 5%, 10 labeld 
    labeled_list = read_list('labeled_5p')
    unlabeled_list = read_list('unlabeled_5p')
else:
    print('Error labelnum!')
    os.exit()
    
eval_list = read_list('eval')
test_list = read_list('test')

if args.GA:
    snapshot_path = args.save_path + "/{}_{}_GA_{}labeled".format(args.dataset_name, args.exp, args.labelnum)
else:
    snapshot_path = args.save_path + "/{}_{}_{}labeled".format(args.dataset_name, args.exp, args.labelnum)
print('snapshot_path: ', snapshot_path)


num_classes = 16
class_momentum = 0.999
patch_size = (96, 96, 96)

train_data_path = args.root_path
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs
cube_size = args.cube_size


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def config_log(snapshot_path_tmp, typename):

    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(labeled_list, unlabeled_list, eval_list, fold_id=1):
    
    snapshot_path_tmp = snapshot_path
    train_list = labeled_list + unlabeled_list    
    handler, sh = config_log(snapshot_path_tmp, 'fold' + str(fold_id))
    logging.info(str(args))

    # make model, optimizer, and lr scheduler
    model_A = VNet(n_channels=1, n_classes=num_classes).cuda()
    model_B = VNet(n_channels=1, n_classes=num_classes).cuda()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = xavier_normal_init_weight(model_B)

    db_train = AMOS_fast(labeled_list, unlabeled_list,
                    base_dir=train_data_path,
                    transform=transforms.Compose([
                        RandomCrop(patch_size),
                        ToTensor(),
                    ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(len(unlabeled_list)*2))
    unlabeled_idxs = list(range(len(unlabeled_list)*2, len(unlabeled_list)*4))
    print('labeled_list: ', len(labeled_list))
    print('unlabeled_list: ', len(unlabeled_list))
    print('train_list: ', len(train_list))
    print('eval_list: ', len(eval_list))
    print('test_list: ', len(test_list))
    print(min(labeled_idxs), max(labeled_idxs), min(unlabeled_idxs), max(unlabeled_idxs))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer_A = optim.SGD(model_A.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_B = optim.SGD(model_B.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path_tmp)
    logging.info("{} itertations per epoch".format(len(trainloader)))



    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)
    ce_loss_k100 = GACE(k=100, gama=0.5)


    iter_num = 0
    best_dice_avg = 0
    metric_all_cases = None
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    loc_list = None
    dist_logger = cube_utils.OrganClassLogger(num_classes=num_classes)

    for epoch_num in iterator:
        model_A.train()
        model_B.train()
        cps_w = get_current_consistency_weight(epoch_num, max_epoch)
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            labeled_volume_batch = volume_batch[:labeled_bs]



            output_A = model_A(volume_batch)
            output_B = model_B(volume_batch) 


            outputs_A_soft = F.softmax(output_A, dim=1)
            outputs_B_soft = F.softmax(output_B, dim=1)
            max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
            max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()
            label_l = label_batch[:labeled_bs]
            
            
            loss_seg = ce_loss(output_A[:labeled_bs], label_l.unsqueeze(1)) + ce_loss(output_B[:labeled_bs], label_l.unsqueeze(1))
            loss_seg_dice = dice_loss(outputs_A_soft[:labeled_bs], label_l) + dice_loss(outputs_B_soft[:labeled_bs], label_l)
            loss_sup = loss_seg + loss_seg_dice
            
            
            loss_cps = ce_loss_k100(output_A, max_B) + ce_loss_k100(output_B, max_A)
            
            # loss prop
            loss = loss_sup + cps_w * loss_cps
            
            # loss prop
            loss = loss_sup + cps_w * loss_cps 


            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            loss.backward()
            optimizer_A.step()
            optimizer_B.step()



            iter_num = iter_num + 1

            if iter_num % 100 == 0:
                logging.info('Fold {}, iteration {}: loss: {:.3f}, '
                             'loss_sup: {:.3f}, loss_cps: {:f}, cps_w: {:f}'.format(fold_id, iter_num,
                                                       loss,
                                                       loss_sup,
                                                       loss_cps, cps_w))

            if iter_num >= max_iterations:
                iter_n = max_iterations - 1
            else:
                iter_n = iter_num            
            lr_ = base_lr * (1.0 - iter_n / max_iterations) ** 0.9  
            
            for param_group in optimizer_A.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_B.param_groups:
                param_group['lr'] = lr_
            if iter_num % 1000 == 0:
            # if (iter_num <= 10000 and iter_num % 1000 == 0) or (iter_num > 10000 and iter_num % 500 == 0):

                model_A.eval()
                model_B.eval()
                dice_all, std_all, metric_all_cases = test_amos_vnet_AB.validation_all_case_fast(model_A, model_B,
                                                                                    num_classes=num_classes,
                                                                                    base_dir=train_data_path,
                                                                                    image_list=eval_list,
                                                                                    patch_size=patch_size,
                                                                                    stride_xy=90,
                                                                                    stride_z=80)
                dice_avg = dice_all.mean()

                logging.info('iteration {}, '
                             'average DSC: {:.4f}, '
                             'spleen: {:.4f}, '
                             'r.kidney: {:.4f}, '
                             'l.kidney: {:.4f}, '
                             'gallbladder: {:.4f}, '
                             'esophagus: {:.4f}, '
                             'liver: {:.4f}, '
                             'stomach: {:.4f}, '
                             'aorta: {:.4f}, '
                             'inferior vena cava: {:.4f}'
                             'pancreas: {:.4f}, '
                             'right adrenal gland: {:.4f}, '
                             'left adrenal gland: {:.4f}, '
                             'duodenum: {:.4f}, '
                             'bladder: {:.4f}, '
                             'prostate/uterus: {:.4f}'
                             .format(iter_num,
                                     dice_avg,
                                     dice_all[0],
                                     dice_all[1],
                                     dice_all[2],
                                     dice_all[3],
                                     dice_all[4],
                                     dice_all[5],
                                     dice_all[6],
                                     dice_all[7],
                                     dice_all[8],
                                     dice_all[9],
                                     dice_all[10],
                                     dice_all[11],
                                     dice_all[12],
                                     dice_all[13],
                                     dice_all[14]))

                if dice_avg > best_dice_avg:
                    best_dice_avg = dice_avg
                    best_model_path_A = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}_best_A.pth'.format(str(iter_num).zfill(5), str(best_dice_avg)[:8]))
                    torch.save(model_A.state_dict(), best_model_path_A)
                    best_model_path_B = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}_best_B.pth'.format(str(iter_num).zfill(5), str(best_dice_avg)[:8]))
                    torch.save(model_B.state_dict(), best_model_path_B)
                    logging.info("save best model to {}, B".format(best_model_path_A))
                else:
                    save_mode_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}_A.pth'.format(str(iter_num).zfill(5), str(dice_avg)[:8]))
                    torch.save(model_A.state_dict(), save_mode_path)
                    save_mode_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}_B.pth'.format(str(iter_num).zfill(5), str(dice_avg)[:8]))
                    torch.save(model_B.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                
                model_A.train()
                model_B.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
            
    # save_best_path = os.path.join(snapshot_path_tmp, '{}_best_model.pth'.format(args.model))
    # os.system('cp {} {}'.format(best_model_path, save_best_path))
    
    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases, best_model_path_A, best_model_path_B


if __name__ == "__main__":

    import stat
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.chmod(snapshot_path, stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
    if os.path.exists(snapshot_path + '/GALoss'):
        shutil.rmtree(snapshot_path + '/GALoss')
    shutil.copytree('.', snapshot_path + '/GALoss', shutil.ignore_patterns(['.git', '__pycache__']))


    metric_final, best_model_path_A, best_model_path_B = train(labeled_list, unlabeled_list, eval_list)
    
    
    # save_best_path = best_model_path
    # os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    model_A = create_model(n_classes=num_classes)
    model_A.load_state_dict(torch.load(best_model_path_A))
    model_A.eval()
    model_B = create_model(n_classes=num_classes)
    model_B.load_state_dict(torch.load(best_model_path_B))
    model_B.eval()
    _, _, metric_final = test_amos_vnet_AB.validation_all_case(model_A, model_B, num_classes=num_classes, base_dir=train_data_path, image_list=test_list, patch_size=patch_size, stride_xy=32, stride_z=16)

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_save_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_save_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}, \n'
                 'spleen: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'r.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'l.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'gallbladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'esophagus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'liver: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'stomach: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'aorta: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'ivc: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'pancreas: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'right adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'Left adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'duodenum: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'bladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'prostate/uterus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean(),
                         metric_mean[0][0], metric_std[0][0], metric_mean[1][0], metric_std[1][0], metric_mean[2][0], metric_std[2][0], metric_mean[3][0], metric_std[3][0],
                         metric_mean[0][1], metric_std[0][1], metric_mean[1][1], metric_std[1][1], metric_mean[2][1], metric_std[2][1], metric_mean[3][1], metric_std[3][1],
                         metric_mean[0][2], metric_std[0][2], metric_mean[1][2], metric_std[1][2], metric_mean[2][2], metric_std[2][2], metric_mean[3][2], metric_std[3][2],
                         metric_mean[0][3], metric_std[0][3], metric_mean[1][3], metric_std[1][3], metric_mean[2][3], metric_std[2][3], metric_mean[3][3], metric_std[3][3],
                         metric_mean[0][4], metric_std[0][4], metric_mean[1][4], metric_std[1][4], metric_mean[2][4], metric_std[2][4], metric_mean[3][4], metric_std[3][4],
                         metric_mean[0][5], metric_std[0][5], metric_mean[1][5], metric_std[1][5], metric_mean[2][5], metric_std[2][5], metric_mean[3][5], metric_std[3][5],
                         metric_mean[0][6], metric_std[0][6], metric_mean[1][6], metric_std[1][6], metric_mean[2][6], metric_std[2][6], metric_mean[3][6], metric_std[3][6],
                         metric_mean[0][7], metric_std[0][7], metric_mean[1][7], metric_std[1][7], metric_mean[2][7], metric_std[2][7], metric_mean[3][7], metric_std[3][7],
                         metric_mean[0][8], metric_std[0][8], metric_mean[1][8], metric_std[1][8], metric_mean[2][8], metric_std[2][8], metric_mean[3][8], metric_std[3][8],
                         metric_mean[0][9], metric_std[0][9], metric_mean[1][9], metric_std[1][9], metric_mean[2][9], metric_std[2][9], metric_mean[3][9], metric_std[3][9],
                         metric_mean[0][10], metric_std[0][10], metric_mean[1][10], metric_std[1][10], metric_mean[2][10], metric_std[2][10], metric_mean[3][10], metric_std[3][10],
                         metric_mean[0][11], metric_std[0][11], metric_mean[1][11], metric_std[1][11], metric_mean[2][11], metric_std[2][11], metric_mean[3][11], metric_std[3][11],
                         metric_mean[0][12], metric_std[0][12], metric_mean[1][12], metric_std[1][12], metric_mean[2][12], metric_std[2][12], metric_mean[3][12], metric_std[3][12],
                         metric_mean[0][13], metric_std[0][13], metric_mean[1][13], metric_std[1][13], metric_mean[2][13], metric_std[2][13], metric_mean[3][13], metric_std[3][13],
                         metric_mean[0][14], metric_std[0][14], metric_mean[1][14], metric_std[1][14], metric_mean[2][14], metric_std[2][14], metric_mean[3][14], metric_std[3][14]))

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
