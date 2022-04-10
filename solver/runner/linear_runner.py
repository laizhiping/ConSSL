import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from thop import profile, clever_format
from tqdm import tqdm

from solver.utils import data_reader
from solver.model import linear

class Trainer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.task = args.task
        self.feature_dim = args.feature_dim
        self.temperature = args.temperature
        self.k = args.k
        self.model_dir = args.model_dir
        self.dataset_name = args.dataset_name
        self.dataset_path = args.dataset_path


        self.gestures = args.gestures
        self.num_channels = args.num_channels
        self.num_gestures  = len(self.gestures)

        self.window_size = args.train["window_size"]
        self.window_step = args.train["window_step"]
        self.batch_size = args.train["batch_size"]
        self.num_epochs = args.train["num_epochs"]

        self.subjects = args.train["subjects"]
        self.train_sessions = args.train["train_sessions"]
        self.test_sessions = args.train["test_sessions"]
        self.train_trials = args.train["train_trials"]
        self.test_trials = args.train["test_trials"]
    
    def get_data_loader(self, subjects, sessions, gestures, trials):
        dataset = data_reader.DataReader(subjects, sessions, gestures, trials, self.window_size, self.window_step, self.dataset_name, self.dataset_path)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=0)
        return data_loader



    # train or test for one epoch
    def train_val(self, epoch, net, data_loader, train_optimizer, loss_criterion):
        is_train = train_optimizer is not None
        net.train() if is_train else net.eval()

        total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
        with (torch.enable_grad() if is_train else torch.no_grad()):
            for data, target in data_bar:
                data, target = data.float().cuda(non_blocking=True), target.cuda(non_blocking=True)
                out = net(data)
                loss = loss_criterion(out, target)

                if is_train:
                    train_optimizer.zero_grad()
                    loss.backward()
                    train_optimizer.step()

                total_num += data.size(0)
                total_loss += loss.item() * data.size(0)
                prediction = torch.argsort(out, dim=-1, descending=True)
                total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                        .format('Train' if is_train else 'Test', epoch, self.num_epochs, total_loss / total_num,
                                                total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

        return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


    def leave_one_out(self):
        acc = np.zeros((len(self.subjects), 2))
        for i, subject in enumerate(self.subjects):
            if self.args.task == "inter_session":
                pass
            elif self.args.task == "inter_subject":
                train_loader = self.get_data_loader([subject], self.train_sessions, self.gestures, self.train_trials)
                test_loader = self.get_data_loader([subject], self.test_sessions, self.gestures, self.test_trials)

            results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
                    'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
            save_name_pre = '{}_{}_{}_{}_{}_{}'.format(subject, self.num_epochs, self.batch_size, self.feature_dim, self.temperature, self.k)
            save_dir = "{}/{}/{}".format(self.model_dir, self.task, self.dataset_name)
            model = linear.Net(self.num_channels, self.window_size, self.num_gestures, pretrained_path=f'{save_dir}/{subject}_pretrained_model.pth').cuda()
            for param in model.f.parameters():
                param.requires_grad = False

            # flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).cuda(),))
            # flops, params = clever_format([flops, params])
            # print('# Model Params: {} FLOPs: {}'.format(params, flops))
            optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
            loss_criterion = nn.CrossEntropyLoss()
            best_acc1, best_acc5 = 0.0, 0.0
            for epoch in range(1, self.num_epochs + 1):
                train_loss, train_acc_1, train_acc_5 = self.train_val(epoch, model, train_loader, optimizer, loss_criterion)
                results['train_loss'].append(train_loss)
                results['train_acc@1'].append(train_acc_1)
                results['train_acc@5'].append(train_acc_5)
                test_loss, test_acc_1, test_acc_5 = self.train_val(epoch, model, test_loader, None, loss_criterion)
                results['test_loss'].append(test_loss)
                results['test_acc@1'].append(test_acc_1)
                results['test_acc@5'].append(test_acc_5)
                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
                data_frame.to_csv('{}/{}_linear_statistics.csv'.format(save_dir, save_name_pre), index_label='epoch')
                if test_acc_1 > best_acc1:
                    best_acc1, best_acc5 = test_acc_1, test_acc_5
                    torch.save(model.state_dict(), '{}/{}_linear_model.pth'.format(save_dir, subject))
            
            acc[i][0], acc[i][1] = best_acc1, best_acc5
            self.logger.info(f"Subject {subject} Acc@1 Acc@5: {best_acc1} {best_acc5}")
            
        self.logger.info(f"All Acc@1 Acc@5 :\n{acc}")
        self.logger.info(f"Avg Acc@1 Acc@5:\n {np.mean(acc, axis=0)}")

    def train_test_split(self):

        acc = np.zeros((len(self.subjects), 2))
        for i, subject in enumerate(self.subjects):
            if self.args.task == "inter_session":
                pass
            elif self.args.task == "inter_subject":
                train_loader = self.get_data_loader([subject], self.train_sessions, self.gestures, self.train_trials)
                test_loader = self.get_data_loader([subject], self.test_sessions, self.gestures, self.test_trials)

            results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
                    'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
            save_name_pre = '{}_{}_{}_{}_{}_{}'.format(subject, self.num_epochs, self.batch_size, self.feature_dim, self.temperature, self.k)
            save_dir = "{}/{}/{}".format(self.model_dir, self.task, self.dataset_name)
            model = linear.Net(self.num_channels, self.window_size, self.num_gestures, pretrained_path=f'{save_dir}/pretrained_model.pth').cuda()
            for param in model.f.parameters():
                param.requires_grad = False

            # flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).cuda(),))
            # flops, params = clever_format([flops, params])
            # print('# Model Params: {} FLOPs: {}'.format(params, flops))
            optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
            loss_criterion = nn.CrossEntropyLoss()
            best_acc1, best_acc5 = 0.0, 0.0
            for epoch in range(1, self.num_epochs + 1):
                train_loss, train_acc_1, train_acc_5 = self.train_val(epoch, model, train_loader, optimizer, loss_criterion)
                results['train_loss'].append(train_loss)
                results['train_acc@1'].append(train_acc_1)
                results['train_acc@5'].append(train_acc_5)
                test_loss, test_acc_1, test_acc_5 = self.train_val(epoch, model, test_loader, None, loss_criterion)
                results['test_loss'].append(test_loss)
                results['test_acc@1'].append(test_acc_1)
                results['test_acc@5'].append(test_acc_5)
                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
                data_frame.to_csv('{}/{}_linear_statistics.csv'.format(save_dir, save_name_pre), index_label='epoch')
                if test_acc_1 > best_acc1:
                    best_acc1, best_acc5 = test_acc_1, test_acc_5
                    torch.save(model.state_dict(), '{}/{}_linear_model.pth'.format(save_dir, subject))
            
            acc[i][0], acc[i][1] = best_acc1, best_acc5
            self.logger.info(f"Subject {subject} Acc@1 Acc@5: {best_acc1} {best_acc5}")
        self.logger.info(f"All Acc@1 Acc@5 :\n{acc}")
        self.logger.info(f"Avg Acc@1 Acc@5:\n {np.mean(acc, axis=0)}")


    def test(self):
        subjects = self.args.subjects
        gestures = self.args.gestures
        num_channels = self.args.num_channels
        num_gestures  = len(gestures)
        sessions = self.args.sessions
        test_trials = self.args.test_trials
        if self.args.task == "inter-session":
            test_sessions = self.args.test_sessions

        window_size =self.args.window_size

        acc_1 = np.zeros(len(subjects))
        acc_5 = np.zeros(len(subjects))
        for i, subject in enumerate(subjects):
            self.logger.info(f"Begin test linear: subject {subject}")
            if self.args.task == "inter-session":
                test_loader = self.get_data_loader([subject], test_sessions, gestures, test_trials)
            elif self.args.task == "inter_subject":
                test_loader = self.get_data_loader([subject], sessions, gestures, test_trials)

            save_dir = "{}/{}/{}".format(self.args.model_dir, self.args.task, self.args.dataset_name)
            model = linear.Net(num_channels, window_size, num_gestures, pretrained_path=f'{save_dir}/{subject}_linear_model.pth').cuda()
            for param in model.f.parameters():
                param.requires_grad = False

            # flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).cuda(),))
            # flops, params = clever_format([flops, params])
            # print('# Model Params: {} FLOPs: {}'.format(params, flops))
            loss_criterion = nn.CrossEntropyLoss()
            test_loss, test_acc_1, test_acc_5 = self.train_val(1, model, test_loader, None, loss_criterion)
            acc_1[i] = test_acc_1
            acc_5[i] = test_acc_5
        
        print(f"ACC@1:")
        for item in acc_1:
            print(item)
        print(f"\nACC@5:")
        for item in acc_5:
            print(item)
        print(f"Mean Acc@1: {acc_1.mean()}, mean Acc@5: {acc_5.mean()}") 