import argparse
from torch_geometric.data import InMemoryDataset
from SpatSite import *
import sys
import os
import time
import datetime
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torchnet import meter
from metrics_calculation import *
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    # dataset
    parser.add_argument("--dataset_name", dest="dataset_name", default='example', help="Type of ligand.")
    parser.add_argument("--train_name", dest='train_name', default='Spat_train', help='Name of train datasets.')
    parser.add_argument("--valid_name", dest='valid_name', default='Spat_valid', help='Name of valid datasets.')
    parser.add_argument("--test_name", dest='test_name', default='Spat_test', help='Name of test datasets.')
    # output size
    parser.add_argument("--res_hidden_size", dest='res_hidden_size', type=int, default=128, help='The dimension of encoded residue feature vector.')
    parser.add_argument("--gra_hidden_size", dest='gra_hidden_size', type=int, default=128, help='The dimension of encoded graph feature vector.')
    parser.add_argument("--str_hidden_size", dest='str_hidden_size', type=int, default=128, help='The dimension of encoded structure feature vector.')
    parser.add_argument("--edg_hidden_size", dest='edg_hidden_size', type=int, default=128, help='The dimension of encoded edge feature vector.')
    # model parameter
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=64, help='Batch size for training deep model.')
    parser.add_argument("--lr", dest='lr', type=float, default=0.0001, help='Learning rate for training the deep model.')
    parser.add_argument("--num_update", dest='num_update', type=int, default=1, help='The number of update')
    parser.add_argument("--num_heads", dest='num_heads', type=int, default=4, help='The number of heads')
    # train parameter
    parser.add_argument("--epoch", dest='epoch', type=int, default=30, help='Training epochs.')
    parser.add_argument("--stop_epoch", dest='stop_epoch', type=int, default=10, help='early_stop_epochs.')
    # topological parameter
    parser.add_argument("--context_radius", dest='context_radius', type=int, default=20, help='context_radius.')
    return parser.parse_args()
def checkargs(args):
    if args.dataset_name is None:
        print('ERROR (dataset_name): Please input the dataset name!')
        raise ValueError
    return
class LoadData_pt(InMemoryDataset):
    def __init__(self, root, dataset):
        super(LoadData_pt, self).__init__(root)
        self.data, self.slices = torch.load(root + '/' + dataset + '.pt')

class Config():
    def __init__(self, args):
        # dataset
        self.train_name = args.train_name
        self.valid_name = args.valid_name
        self.test_name = args.test_name
        self.str_dataio = LoadData_pt
        # model parameter
        self.str_model = SpatSite_Model
        self.max_metric = 'F1'
        # # output
        self.x_hs = args.res_hidden_size
        self.u_hs = args.gra_hidden_size
        self.t_hs = args.str_hidden_size
        self.e_hs = args.edg_hidden_size
        # # model parameter
        self.num_update = args.num_update
        self.dropratio = 0.1
        self.str_lr = args.lr
        self.L2_weight = 0
        self.bias = True
        self.num_workers = 0
        # Train and Test parameter
        self.batch_size = args.batch_size
        self.test_batchsize = args.batch_size
        self.epoch = args.epoch
        self.early_stop_epochs = args.stop_epoch
        self.max_metric_th = 'F1' #选择th指标
        # topological parameter
        self.context_radius = args.context_radius
        # # PATH
        self.dataset_name = args.dataset_name
        self.Dataset_dir = os.path.abspath('..') + '/data/' + self.dataset_name
        model_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.model_time = model_time
        self.output_folder = os.path.abspath('..') + '/checkpoints/' + self.dataset_name + '_train'
        if not os.path.exists(self.output_folder): os.makedirs(self.output_folder)

    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass

def data_report(train_data, valid_data, test_data):
    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'NumRes', 'Pos', 'Neg', 'PNratio']
    tb.float_format = "2.3"

    Numres = train_data.data.y.shape[0]
    pos = torch.sum(train_data.data.y).item()
    neg = train_data.data.y.shape[0] - pos
    tb.add_row(['train', Numres, int(pos), int(neg), pos / float(neg)])

    Numres = valid_data.data.y.shape[0]
    pos = torch.sum(valid_data.data.y).item()
    neg = valid_data.data.y.shape[0] - pos
    tb.add_row(['valid', Numres, int(pos), int(neg), pos / float(neg)])

    Numres = test_data.data.y.shape[0]
    pos = torch.sum(test_data.data.y).item()
    neg = test_data.data.y.shape[0] - pos
    tb.add_row(['test', Numres, int(pos), int(neg), pos / float(neg)])

    print(tb)

def val(opt, device, model, valid_data, dataset_type, epoch, val_th=None):
    valid_dataloader = DataLoader(valid_data, batch_size=opt.test_batchsize, shuffle=False,
                              num_workers=opt.num_workers, pin_memory=True)
    model.eval()
    if val_th is not None:
        AUC_meter = meter.AUCMeter()
        Confusion_meter = meter.ConfusionMeter(k=2)
        y_true = []
        y_scores = []
        with torch.no_grad():
            for ii, data in enumerate(valid_dataloader):
                data = data.to(device)
                target = data.y
                score = model(data, device).float()
                y_true.extend(target.cpu().numpy())
                y_scores.extend(score.cpu().numpy())
                AUC_meter.add(score, target)
                pred_bi = target.data.new(score.shape).fill_(0)
                pred_bi[score > val_th] = 1
                Confusion_meter.add(pred_bi, target)
        val_auc = AUC_meter.value()[0]
        cfm = Confusion_meter.value()
        val_rec, val_pre, val_F1, val_spe, val_mcc = CFM_eval_metrics(cfm)
        val_prc = average_precision_score(y_true, y_scores)
    else:
        AUC_meter = meter.AUCMeter()
        for j in range(2,100,2):
            th = j/100.0
            locals()['Confusion_meter_' + str(th)] = meter.ConfusionMeter(k=2)
        y_true = []
        y_scores = []
        with torch.no_grad():
            for ii, data in enumerate(valid_dataloader):
                data = data.to(device)
                target = data.y
                score = model(data, device).float()
                y_true.extend(target.cpu().numpy())
                y_scores.extend(score.cpu().numpy())
                AUC_meter.add(score, target)
                for j in range(2, 100, 2):
                    th = j / 100.0
                    pred_bi = target.data.new(score.shape).fill_(0)
                    pred_bi[score > th] = 1
                    locals()['Confusion_meter_' + str(th)].add(pred_bi,target)
        val_auc = AUC_meter.value()[0]
        val_prc = average_precision_score(y_true, y_scores)
        val_rec, val_pre, val_F1, val_spe, val_mcc = -1, -1, -1, -1, -1
        val_th = 0
        for j in range(2, 100, 2):
            th = j / 100.0
            cfm = locals()['Confusion_meter_' + str(th)].value()
            rec, pre, F1, spe, mcc = CFM_eval_metrics(cfm)
            if opt.max_metric_th == 'MCC':
                if mcc > val_mcc:
                    val_rec, val_pre, val_F1, val_spe, val_mcc, val_th = rec, pre, F1, spe, mcc, th
            elif opt.max_metric_th == 'F1':
                if F1 >= val_F1:
                    val_rec, val_pre, val_F1, val_spe, val_mcc, val_th = rec, pre, F1, spe, mcc, th
            else:
                print('ERROR: opt.max_metric.')
                raise ValueError
    print("{}_epoch_{} result: th = {:.2f}; recall = {:.3f}; precision = {:.3f}; "
          "F1 score = {:.3f}; spe = {:.3f}; MCC = {:.3f}; AUC = {:.3f}; PRC = {:.3f}"
          .format(dataset_type, epoch+1, val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc))

    return val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc

def test(opt, device, test_data):
    model_path = '{}/{}_spatsite_model.pth'.format(opt.output_folder, opt.dataset_name)
    model, criterion, optimizer, th, _ = torch.load(model_path)
    model.to(device)
    model.eval()
    test_dataloader = DataLoader(test_data, batch_size=opt.test_batchsize, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for ii, data in enumerate(test_dataloader):
            data = data.to(device)
            target = data.y
            score = model(data, device).float()
            test_probs += score.tolist()
            test_targets += target.tolist()
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    return test_probs, test_targets

def train(opt, device, model, learning_rate, train_data, valid_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=opt.L2_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6,
                                                           patience=10, min_lr=1e-6)
    criterion = nn.BCELoss()
    model.to(device)
    criterion.to(device)
    loss_meter = meter.AverageValueMeter()
    early_stop_iter = 0
    max_metric_val = -2
    epoch_begin = 0

    for epoch in range(epoch_begin, opt.epoch):
        print("\n!!!!!!!!!!!!!!!!!! epoch:{} / {} !!!!!!!!!!!!!!!!!!".format(epoch+1,opt.epoch))
        epoch_start_time = datetime.datetime.now()  ### cancel
        nstep = len(train_dataloader)
        metrice_val = 0
        for ii, data in enumerate(train_dataloader):
            model.train()
            data = data.to(device)
            target = data.y
            optimizer.zero_grad()
            score = model(data, device)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            if ii % (nstep - 1) == 0 and ii != 0:
                print('|| Epoch{} step{} || lr={:.6f} | train_loss={:.5f}'.format(epoch, ii,
                                                                                  optimizer.param_groups[0]['lr'],
                                                                                  loss_meter.mean))
                print('------------ Valid START ------------')
                val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc = val(opt, device, model, valid_data, 'valid', epoch)
                _ = val(opt, device, model, test_data, 'test', epoch, val_th)
                if opt.max_metric == 'AUC':
                    metrice_val = val_auc
                elif opt.max_metric == 'MCC':
                    metrice_val = val_mcc
                elif opt.max_metric == 'F1':
                    metrice_val = val_F1
                elif opt.max_metric == 'PRC':
                    metrice_val = val_prc
                else:
                    print('ERROR: opt.max_metric.')
                    raise ValueError
                if metrice_val > max_metric_val:
                    print('------------ SAVE START ------------')
                    max_metric_val = metrice_val
                    early_stop_iter = 0
                    save_path = '{}/{}_spatsite_model.pth'.format(opt.output_folder, opt.dataset_name)
                    print('save net', save_path)
                    torch.save([model, criterion, optimizer, val_th, epoch], save_path)
                else:
                    early_stop_iter += 1
        scheduler.step(metrice_val)
        loss_meter.reset()
        epoch_end_time = datetime.datetime.now()
        print('Epoch elapsed time: %s ' % (epoch_end_time - epoch_start_time))
        if early_stop_iter >= opt.early_stop_epochs:
            break
    return

def main(opt,device):
    print("!!!START!!!")
    print('=' * 89)
    print('||parameter||')
    opt.print_config()
    print('device = {}'.format(device))
    print('=' * 40 + 'structure' + '=' * 40)

    train_data = opt.str_dataio(root=opt.Dataset_dir, dataset=opt.train_name)
    valid_data = opt.str_dataio(root=opt.Dataset_dir, dataset=opt.valid_name)
    test_data = opt.str_dataio(root=opt.Dataset_dir, dataset=opt.test_name)

    data_report(train_data, valid_data, test_data)
    x_ind = train_data._data.x.shape[1]
    ex_ind = train_data._data.ex.shape[1] if 'ex' in train_data.slices.keys() else 0
    model = opt.str_model(lx_ind=x_ind, hx_ind=ex_ind, x_hs=opt.x_hs, u_hs=opt.u_hs, t_hs=opt.t_hs, e_hs=opt.e_hs,
                          dropratio=opt.dropratio, bias=opt.bias, num_update=opt.num_update,
                          context_radius=opt.context_radius)
    learning_rate = opt.str_lr

    print(model)  ### cancel
    train_time = datetime.datetime.now()  ### cancel
    print('!!!!!!!!!!!!!!!! TRAINing START !!!!!!!!!!!!!!!!')
    train(opt, device, model, learning_rate, train_data, valid_data, test_data)
    print('Training time: %s seconds' % (datetime.datetime.now() - train_time).seconds)  ### cancel
    print('!!!!!!!!!!!!!!!! TESTing START !!!!!!!!!!!!!!!!')
    valid_probs, valid_labels = test(opt, device, valid_data)
    test_probs, test_labels = test(opt, device, test_data)
    results = {'valid_probs': valid_probs, 'valid_labels': valid_labels,
               'test_probs': test_probs, 'test_labels': test_labels}
    with open(opt.output_folder+'/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, prc_ = eval_metrics(valid_probs, valid_labels, opt.max_metric_th)
    valid_matrices = th_, rec_, pre_, f1_, spe_, mcc_, auc_, prc_
    th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, prc_ = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, rec_, pre_, f1_, spe_, mcc_, auc_, prc_
    print_results(valid_matrices, test_matrices)
    return

if __name__ == '__main__':
    args = parse_args()
    checkargs(args)
    opt = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.stdout = Logger(opt.output_folder + '/' + opt.model_time + '.log')
    START_time = datetime.datetime.now()  ### cancel

    main(opt, device)

    END_time = datetime.datetime.now()  ### cancel
    print('Total elapsed time: %s seconds' % (END_time - START_time).seconds)  ### cancel
    sys.stdout.log.close()