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
from utils import metrics, ramps, test_amos, cube_losses, cube_utils
from dataloaders.dataset import *
from networks.magicnet import VNet_Magic
from GALoss import GADice, GACE


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='AMOS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./data/AMOS/', help='Name of Dataset')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save')
parser.add_argument('--exp', type=str, default='MagicNet', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=17000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=30, help='total samples of the dataset')
parser.add_argument('--max_train_samples', type=int, default=20, help='maximum samples to train')
parser.add_argument('--max_test_samples', type=int, default=6, help='maximum samples to test')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=4, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)


def create_model(n_classes=14, cube_size=32, patchsize=96, ema=False):
    # Network definition
    net = VNet_Magic(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patchsize)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
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
print(snapshot_path)

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

    model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0])
    ema_model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0], ema=True)

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

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr)

    writer = SummaryWriter(snapshot_path_tmp)
    logging.info("{} itertations per epoch".format(len(trainloader)))




    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)      


    ema_model.train()

    iter_num = 0
    best_dice_avg = 0
    metric_all_cases = None
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    loc_list = None
    dist_logger = cube_utils.OrganClassLogger(num_classes=num_classes)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            labeled_volume_batch = volume_batch[:labeled_bs]

            model.train()
            outputs = model(volume_batch)[0] # Original Model Outputs

            # Cross-image Partition-and-Recovery
            bs, c, w, h, d = volume_batch.shape
            nb_cubes = h // cube_size
            cube_part_ind, cube_rec_ind = cube_utils.get_part_and_rec_ind(volume_shape=volume_batch.shape,
                                                                          nb_cubes=nb_cubes,
                                                                          nb_chnls=16)
            img_cross_mix = volume_batch.view(bs, c, w, h, d)
            img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
            img_cross_mix = img_cross_mix.view(bs, c, w, h, d)
            

            outputs_mix, embedding = model(img_cross_mix)
            c_ = embedding.shape[1]
            pred_rec = torch.gather(embedding, dim=0, index=cube_rec_ind)
            pred_rec = pred_rec.view(bs, c_, w, h, d)
            outputs_unmix = model.forward_prediction_head(pred_rec)
            

            # Get pseudo-label from teacher model
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)[0]
                unlab_pl_soft = F.softmax(ema_output, dim=1)
                pred_value_teacher, pred_class_teacher = torch.max(unlab_pl_soft, dim=1)

            # nt = 3, ts = 32
            # loc_list: 27 x [1, 1] (x + Wy + WHz)
            if iter_num == 0:
                loc_list = cube_utils.get_loc_mask(volume_batch, cube_size)

            # calculate some losses
            loss_seg = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            outputs_soft = F.softmax(outputs, dim=1)
            outputs_unmix_soft = F.softmax(outputs_unmix, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs])
            loss_unmix_dice = dice_loss(outputs_unmix_soft[:labeled_bs], label_batch[:labeled_bs])
            
            
            supervised_loss = (loss_seg + loss_seg_dice + loss_unmix_dice)
            count_ss = 3

            # Magic-cube Location Reasoning
            # patch_list: N=27 x [4, 1, 1, 32, 32, 32] (bs, pn, c, w, h, d)
            patch_list = cube_losses.get_patch_list(volume_batch, cube_size=cube_size)
            # idx = 27
            idx = torch.randperm(len(patch_list)).cuda()  # random
            # cube location loss
            loc_loss = 0
            feat_list = None
            if loc_list is not None:
                loc_loss, feat_list = cube_losses.cube_location_loss(model, loc_list, patch_list, idx, labeled_bs, cube_size=cube_size)

            consistency_loss = 0
            count_consist = 1

            # Within-image Partition-and-Recovery
            if feat_list is not None:
                embed_list = []
                for i in range(bs):
                    pred_tmp, embed_tmp = model.forward_decoder(feat_list[i])
                    embed_list.append(embed_tmp.unsqueeze(0))

                embed_all = torch.cat(embed_list, dim=0)
                embed_all_unmix = cube_losses.unmix_tensor(embed_all, labeled_volume_batch.shape)
                pred_all_unmix = model.forward_prediction_head(embed_all_unmix) 
                unmix_pred_soft = F.softmax(pred_all_unmix, dim=1)
                loss_lab_local_dice = dice_loss(unmix_pred_soft[:labeled_bs], label_batch[:labeled_bs])
                supervised_loss += loss_lab_local_dice
                count_ss += 1
                

            # Cube-wise Pseudo-label Blending
            pred_class_mix = None
            with torch.no_grad():
                # To store some class pixels at the beginning of training to calculate the organ-class dist
                if iter_num > 100 and feat_list is not None:
                    # Get organ-class distribution
                    current_organ_dist = dist_logger.get_class_dist().cuda()  # (1, C)
                    # Normalize
                    current_organ_dist = current_organ_dist ** (1. / args.T_dist)
                    current_organ_dist = current_organ_dist / current_organ_dist.sum()
                    current_organ_dist = current_organ_dist / current_organ_dist.max()


                    weight_map = current_organ_dist[pred_class_teacher].unsqueeze(1).repeat(1, num_classes, 1, 1, 1)

                    unmix_pl = cube_losses.get_mix_pl(model, feat_list, volume_batch.shape, bs - labeled_bs)
                    unlab_pl_mix = (1. - weight_map) * ema_output + weight_map * unmix_pl
                    unlab_pl_mix_soft = F.softmax(unlab_pl_mix, dim=1)
                    _, pred_class_mix = torch.max(unlab_pl_mix_soft, dim=1)

                    # pr_class: 2x96**3, 1
                    conf, pr_class = torch.max(unlab_pl_mix_soft.detach(), dim=1)
                    dist_logger.append_class_list(pr_class.view(-1, 1))

                elif feat_list is not None:
                    conf, pr_class = torch.max(unlab_pl_soft.detach(), dim=1)
                    dist_logger.append_class_list(pr_class.view(-1, 1))


            if iter_num % 20 == 0 and len(dist_logger.class_total_pixel_store):
                dist_logger.update_class_dist()

            consistency_weight = get_current_consistency_weight(iter_num // 350)
            # debiase the pseudo-label: blend ema and unmixed_within pseudo-label
            if pred_class_mix is None:
                consistency_loss_unmix = dice_loss(outputs_unmix_soft[labeled_bs:], pred_class_teacher)
            else:
                consistency_loss_unmix = dice_loss(outputs_unmix_soft[labeled_bs:], pred_class_mix)

            consistency_loss += consistency_loss_unmix

            supervised_loss /= count_ss
            consistency_loss /= count_consist

            # Final Loss
            loss = supervised_loss + 0.1 * loc_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            if iter_num % 100 == 0:
                logging.info('Fold {}, iteration {}: loss: {:.4f}, '
                             'cons_dist: {:.4f}, loss_weight: {:4f}, '
                             'loss_loc: {:.4f}'.format(fold_id, iter_num,
                                                       loss,
                                                       consistency_loss,
                                                       consistency_weight,
                                                       0.1 * loc_loss))
            lr_ = base_lr * (1.0 - iter_num / 70000) ** 0.9 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            if iter_num % 1000 == 0:

                model.eval()
                dice_all, std_all, metric_all_cases = test_amos.validation_all_case_fast(model,
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
                    best_model_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}_best.pth'.format(str(iter_num).zfill(5), str(best_dice_avg)[:8]))
                    torch.save(model.state_dict(), best_model_path)
                    logging.info("save best model to {}".format(best_model_path))
                else:
                    save_mode_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}.pth'.format(str(iter_num).zfill(5), str(dice_avg)[:8]))
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                
                model.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
    
    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases, best_model_path


if __name__ == "__main__":

    import stat
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.chmod(snapshot_path, stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
    if os.path.exists(snapshot_path + '/GALoss'):
        shutil.rmtree(snapshot_path + '/GALoss')
    shutil.copytree('.', snapshot_path + '/GALoss', shutil.ignore_patterns(['.git', '__pycache__']))


    metric_final, best_model_path = train(labeled_list, unlabeled_list, eval_list)
    
    save_best_path = best_model_path
    model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0])
    model.load_state_dict(torch.load(save_best_path))
    model.eval()
    _, _, metric_final = test_amos.validation_all_case(model, num_classes=num_classes, base_dir=train_data_path, image_list=test_list, patch_size=patch_size, stride_xy=32, stride_z=16)

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
