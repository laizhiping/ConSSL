import os
import torch
import pandas as pd
import numpy as np
from thop import profile, clever_format
from tqdm import tqdm

from solver.utils import data_reader
from solver.model import framework

class Trainer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
    
    def get_pair_loader(self, subjects, sessions, gestures, trials):
        dataset = data_reader.PairReader(subjects, sessions, gestures, trials, self.args)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=0)
        return data_loader
    
    # train for one epoch to learn unique features
    def train(self, epoch, net, data_loader, train_optimizer):
        net.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        for pos_1, pos_2, target in train_bar:
            pos_1, pos_2 = pos_1.float().cuda(non_blocking=True), pos_2.float().cuda(non_blocking=True)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.args.temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += self.args.batch_size
            total_loss += loss.item() * self.args.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.args.num_epochs, total_loss / total_num))

        return total_loss / total_num


    # test for one epoch, use weighted knn to find the most similar images' label to assign the test image
    def test(self, epoch, net, memory_data_loader, test_data_loader):
        net.eval()
        total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            # generate feature bank
            for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out = net(data.float().cuda(non_blocking=True))
                feature_bank.append(feature)
                feature_labels.append(target)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
            feature_labels = torch.cat(feature_labels, dim=0).cuda(non_blocking=True)

            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(test_data_loader)
            for data, _, target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature, out = net(data.float().cuda(non_blocking=True))

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=self.args.k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / self.args.temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * self.args.k, len(self.args.gestures), device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, len(self.args.gestures)) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                        .format(epoch, self.args.num_epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

        return total_top1 / total_num * 100, total_top5 / total_num * 100

    def start(self):
        subjects = self.args.subjects
        gestures = self.args.gestures
        num_channels = self.args.num_channels
        num_gestures  = len(gestures)
        trials = self.args.trials
        train_sessions = self.args.train_sessions
        test_sessions = self.args.test_sessions
        train_trials = self.args.train_trials
        test_trials = self.args.test_trials


        accuracy = np.zeros(len(subjects))
        for i, subject in enumerate(subjects):
            self.logger.info(f"Begin pretraining subject {subject}")
            # 可能需要改动
            train_loader = self.get_pair_loader([subject], train_sessions, gestures, trials)
            memory_loader = self.get_pair_loader([subject], train_sessions, gestures, trials)
            test_loader = self.get_pair_loader([subject], test_sessions, gestures, test_trials)
            # model = self.get_model()
            
            # model setup and optimizer config
            model = framework.Model(self.args.num_channels, num_gestures, feature_dim=128).cuda()
            flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).cuda(),))
            flops, params = clever_format([flops, params])
            print('# Model Params: {} FLOPs: {}'.format(params, flops))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

            
            # training loop
            results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
            save_name_pre = '{}_{}_{}_{}_{}_{}'.format(subject, self.args.feature_dim, self.args.temperature, self.args.k, self.args.batch_size, self.args.num_epochs)
            best_acc = 0.0
            for epoch in range(1, self.args.num_epochs + 1):
                train_loss = self.train(epoch, model, train_loader, optimizer)
                results['train_loss'].append(train_loss)
                test_acc_1, test_acc_5 = self.test(epoch, model, memory_loader, test_loader)
                results['test_acc@1'].append(test_acc_1)
                results['test_acc@5'].append(test_acc_5)
                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
                data_frame.to_csv('{}/{}_statistics.csv'.format(self.args.model_dir, save_name_pre), index_label='epoch')
                if test_acc_1 > best_acc:
                    best_acc = test_acc_1
                    torch.save(model.state_dict(), '{}/{}_model.pth'.format(self.args.model_dir, save_name_pre))
            
            accuracy[i] = best_acc
        self.logger.info(f"All subject average accuracy:\n {accuracy.mean()}")
            
    