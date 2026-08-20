import argparse
from torch_geometric.data import InMemoryDataset
import sys
import os
from datetime import datetime
from torch_geometric.loader import DataLoader
from metrics_calculation import *

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    # dataset
    parser.add_argument("--dataset_name", dest="dataset_name", default='example', help="Type of ligand.")
    parser.add_argument("--test_name", dest='test_name', default='Spat_test', help='Name of test datasets.')
    # model parameter
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=64, help='Batch size for training deep model.')
    # topological parameter
    parser.add_argument("--context_radius", dest='context_radius', type=int, default=20, help='context_radius.')
    # model
    parser.add_argument("--model_path", dest='model_path', type=str, default=None, help='specified model path.')
    return parser.parse_args()

class LoadData_pt(InMemoryDataset):
    def __init__(self, root, dataset):
        super(LoadData_pt, self).__init__(root)
        self.data, self.slices = torch.load(root + '/' + dataset + '.pt')

class Config():
    def __init__(self, args):
        # dataset
        self.predict_name = args.test_name
        self.str_dataio = LoadData_pt
        # parameters
        self.test_batchsize = args.batch_size
        self.num_workers = 0
        # # PATH
        self.dataset_name = args.dataset_name
        self.Dataset_dir = os.path.abspath('..') + '/data/' + self.dataset_name
        self.output_folder = os.path.abspath('..') + '/results/' #+ self.dataset_name + '_test'
        if not os.path.exists(self.output_folder): os.makedirs(self.output_folder)

        self.log_output_folder = os.path.abspath('..') + '/checkpoints/' #+ self.dataset_name + '_test'
        if not os.path.exists(self.log_output_folder): os.makedirs(self.log_output_folder)

        if args.model_path is None:
            model_folder = os.path.abspath('..') + '/checkpoints/' + self.dataset_name + '_train'
            model_name = '{}_spatsite_model.pth'.format(self.dataset_name)
            self.model_path = '{}/{}'.format(model_folder, model_name)
        else:
            self.model_path = args.model_path

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


def main(opt,device):
    print("!!!START!!!")
    print("TIME: {}".format(datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')))
    print('=' * 89)
    print('||parameter||')
    opt.print_config()
    print('device = {}'.format(device))
    print('=' * 89)

    predict_data = opt.str_dataio(root=opt.Dataset_dir, dataset=opt.predict_name)
    model_path = opt.model_path
    model, criterion, optimizer, th, _ = torch.load(model_path)
    model.to(device)
    model.eval()
    predict_dataloader = DataLoader(predict_data, batch_size=opt.test_batchsize, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)
    test_probs = []
    res_type_list = []
    res_num_list = []
    pro_name_list = []
    test_targets = []
    with torch.no_grad():
        for ii, data in enumerate(predict_dataloader):
            data = data.to(device)
            score = model(data, device).float()
            test_probs += score.tolist()
            target = data.y
            test_targets += target.tolist()

            res_type_list += data.res_type
            pro_name_list += data.pro_name
            res_num_list += data.res_num.tolist()

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    res_type_list = np.array(res_type_list)
    pro_name_list = np.array(pro_name_list)
    res_num_list = np.array(res_num_list)
    predict_site = test_probs > th

    pro_list = list(set(pro_name_list))

    th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, prc_ = th_eval_metrics(th, test_probs, test_targets)
    test_matrices = th_, rec_, pre_, f1_, spe_, mcc_, auc_, prc_
    print_results(None, test_matrices)

    with open(opt.output_folder + '/' + opt.dataset_name + '_test.txt', 'w') as file:
        for pro in pro_list:
            file.write(f">{pro}\n")
            temp_flag = (pro_name_list == pro)
            temp_num = res_num_list[temp_flag]
            temp_type = res_type_list[temp_flag]
            temp_site = predict_site[temp_flag]
            AAseq = ''.join(temp_type[temp_num])
            AAsite = ''.join((temp_site[temp_num].astype(np.uint8)).astype(str))
            file.write(f"{AAseq}\n")
            file.write(f"{AAsite}\n")

if __name__ == '__main__':
    args = parse_args()
    opt = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.stdout = Logger(opt.log_output_folder + '/' + opt.dataset_name + '_test.log')
    START_time = datetime.now()  ### cancel

    main(opt, device)

    END_time = datetime.now()  ### cancel
    print('Total elapsed time: %s seconds\n' % (END_time - START_time).seconds)  ### cancel
    sys.stdout.log.close()