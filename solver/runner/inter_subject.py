import torch
import pandas as pd
import numpy as np
from thop import profile, clever_format
from tqdm import tqdm

from solver.utils import data_reader
from solver.model import framework, linear

class Pretrainer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.task = args.task
        self.loocv = args.loocv
        self.feature_dim = args.feature_dim
        self.temperature = args.temperature
        self.k = args.k
        self.model_dir = args.model_dir
        self.dataset_name = args.dataset_name
        self.dataset_path = args.dataset_path


        self.gestures = args.gestures
        self.num_channels = args.num_channels
        self.num_gestures  = len(self.gestures)

        self.batch_size = args.pretrain["batch_size"]
        self.num_epochs = args.pretrain["num_epochs"]

        self.pretrain_subjects = args.pretrain["subjects"]
        self.pretrain_sessions = args.pretrain["sessions"]
        self.pretrain_trials = args.pretrain["trials"]
        self.train_window_size = args.pretrain["window_size"]
        self.train_window_step = args.pretrain["window_step"]

        self.test_subjects = args.train["subjects"]
        self.test_sessions = args.train["test_sessions"]
        self.test_trials = args.train["test_trials"]
        self.test_window_size = args.train["window_size"]
        self.test_window_step = args.train["window_step"]

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    def get_pair_loader(self, subjects, sessions, gestures, trials, window_size, window_step):
        dataset = data_reader.PairReader(subjects, sessions, gestures, trials, window_size, window_step, self.dataset_name, self.dataset_path)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=0)
        return data_loader
    
    def get_data_loader(self, subjects, sessions, gestures, trials, window_size, window_step):
        dataset = data_reader.DataReader(subjects, sessions, gestures, trials, window_size, window_step, self.dataset_name, self.dataset_path)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=0)
        return data_loader
    
    # train for one epoch to learn unique features
    def train(self, epoch, net, data_loader, train_optimizer):
        net.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        for pos_1, pos_2, target in train_bar:
            pos_1, pos_2 = pos_1.float().to(self.device), pos_2.float().to(self.device)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            # print(pos_1.shape, pos_2.shape, out_1.shape, out_2.shape)
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.size(0), device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.size(0), -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.num_epochs, total_loss / total_num))

        return total_loss / total_num


    def get_feature_bank(self, net, memory_data_loader):
        net.eval()
        feature_bank, feature_labels = [], []
        with torch.no_grad():
            # generate feature bank
            for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'): # torch.Size([16, 1, 52, 8]), torch.Size([16])
                feature, out = net(data.float().to(self.device)) # torch.Size([16, 2912]), torch.Size([16, 128])
                feature_bank.append(feature)
                feature_labels.append(target)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous() # torch.Size([2912, 252])
            # [N]
            # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
            feature_labels = torch.cat(feature_labels, dim=0).to(self.device) # torch.Size([252])
            return feature_bank, feature_labels    

    # test for one epoch, use weighted knn to find the most similar images' label to assign the test image
    def test(self, epoch, net, test_data_loader, feature_bank, feature_labels):
        net.eval()
        total_top1, total_top5, total_num = 0.0, 0.0, 0
        # feature_bank, feature_labels = [], []
        with torch.no_grad():
            # # generate feature bank
            # for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            #     feature, out = net(data.float().to(self.device)
            #     feature_bank.append(feature)
            #     feature_labels.append(target)
            # # [D, N]
            # feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # # [N]
            # # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
            # feature_labels = torch.cat(feature_labels, dim=0).to(self.device)

            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(test_data_loader)
            for data, target in test_bar:
                data, target = data.to(self.device), target.to(self.device)
                feature, out = net(data.float().to(self.device))

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / self.temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * self.k, len(self.gestures), device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, len(self.gestures)) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                        .format(epoch, self.num_epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

        return total_top1 / total_num * 100, total_top5 / total_num * 100

    def start(self):
        if self.loocv:
            self.leave_one_out()
        else:
            self.subject_split()

    def leave_one_out(self):
        accuracy = np.zeros(len(self.pretrain_subjects))
        for i, subject in enumerate(self.pretrain_subjects):
            self.logger.info(f"Pretraining subject {subject}")

            train_subjects =  self.pretrain_subjects[:]
            train_subjects.remove(subject)
            test_subjects = [subject]
            train_loader = self.get_pair_loader(train_subjects, self.pretrain_sessions, self.gestures, self.pretrain_trials, self.train_window_size, self.train_window_step)
            memory_loader = self.get_pair_loader(train_subjects, self.pretrain_sessions, self.gestures, self.pretrain_trials, self.train_window_size, self.train_window_step)
            test_loader = self.get_data_loader(test_subjects, self.test_sessions, self.gestures, self.test_trials, self.test_window_size, self.test_window_step)


            # model setup and optimizer config
            model = framework.Model(self.num_channels, self.train_window_size, self.num_gestures, feature_dim=128).to(self.device)
            # flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).to(self.device),))
            # flops, params = clever_format([flops, params])
            # print('# Model Params: {} FLOPs: {}'.format(params, flops))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
            
            feature_bank, feature_labels = self.get_feature_bank(model, memory_loader)
            
            # results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
            results = {'train_loss': []}

            save_name_pre = '00_{}_{}_{}_{}_{}'.format(self.num_epochs, self.batch_size, self.feature_dim, self.temperature, self.k)
            save_dir = "{}/{}/{}".format(self.model_dir, self.task, self.dataset_name)
            best_acc = 0.0
            for epoch in range(1, self.num_epochs + 1):
                train_loss = self.train(epoch, model, train_loader, optimizer)
                results['train_loss'].append(train_loss)
    
                # test_acc_1, test_acc_5 = self.test_subject(epoch, model, feature_bank, feature_labels)
                # results['test_acc@1'].append(test_acc_1)
                # results['test_acc@5'].append(test_acc_5)
                # if test_acc_1 > best_acc:
                #     best_acc = test_acc_1
                #     torch.save(model.state_dict(), '{}/00_model.pth'.format(save_dir))

                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
                data_frame.to_csv('{}/{}_statistics.csv'.format(save_dir, save_name_pre), index_label='epoch')

            test_acc_1, test_acc_5 = self.test(epoch, model, test_loader, feature_bank, feature_labels)
            accuracy[i] = test_acc_1
            self.logger.info(f'Final pretrain accuracy: {test_acc_1}')
            torch.save(model.state_dict(), '{}/{}_pretrained_model.pth'.format(save_dir, subject))

        self.logger.info(f"All subjects: Final pretrain accuracy:\n {accuracy.mean()}")
            
    def subject_split(self):

        train_loader = self.get_pair_loader(self.pretrain_subjects, self.pretrain_sessions, self.gestures, self.pretrain_trials, self.train_window_size, self.train_window_step)
        memory_loader = self.get_pair_loader(self.pretrain_subjects, self.pretrain_sessions, self.gestures, self.pretrain_trials, self.train_window_size, self.train_window_step)
        # test_loader = self.get_data_loader(self.test_subjects, self.test_sessions, self.gestures, self.test_trials, self.test_window_size, self.test_window_step)

        # model setup and optimizer config
        model = framework.Model(self.num_channels, self.train_window_size, self.num_gestures, feature_dim=128).to(self.device)
        # flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).to(self.device),))
        # flops, params = clever_format([flops, params])
        # print('# Model Params: {} FLOPs: {}'.format(params, flops))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        
        feature_bank, feature_labels = self.get_feature_bank(model, memory_loader)
        
        # results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
        results = {'train_loss': []}

        save_name_pre = '00_{}_{}_{}_{}_{}'.format(self.num_epochs, self.batch_size, self.feature_dim, self.temperature, self.k)
        save_dir = "{}/{}/{}".format(self.model_dir, self.task, self.dataset_name)

        best_acc = 0.0
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train(epoch, model, train_loader, optimizer)
            results['train_loss'].append(train_loss)
 
            # test_acc_1, test_acc_5 = self.test_subject(epoch, model, feature_bank, feature_labels)
            # results['test_acc@1'].append(test_acc_1)
            # results['test_acc@5'].append(test_acc_5)
            # if test_acc_1 > best_acc:
            #     best_acc = test_acc_1
            #     torch.save(model.state_dict(), '{}/00_model.pth'.format(save_dir))

            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('{}/{}_statistics.csv'.format(save_dir, save_name_pre), index_label='epoch')

        test_acc_1, test_acc_5 = self.test_subject(self.num_epochs, model, feature_bank, feature_labels)
        self.logger.info(f'Final pretrain accuracy: {test_acc_1}')
        torch.save(model.state_dict(), '{}/pretrained_model.pth'.format(save_dir))

    def test_subject(self, epoch, model, feature_bank, feature_labels):
        acc = np.zeros((len(self.test_subjects), 2))
        for i, subject in enumerate(self.test_subjects):
            test_loader = self.get_data_loader([subject], self.test_sessions, self.gestures, self.test_trials, self.test_window_size, self.test_window_step)
            acc1, acc5 = self.test(epoch, model, test_loader, feature_bank, feature_labels)  
            acc[i][0], acc[i][1] = acc1, acc5
        
        acc1, acc5 = np.mean(acc, axis=0)
        return acc1, acc5


class Trainer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.task = args.task
        self.loocv = args.loocv
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
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
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
                data, target = data.float().to(self.device), target.to(self.device)
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


    def start(self):
        acc = np.zeros((len(self.subjects), 2))
        for i, subject in enumerate(self.subjects):
            train_loader = self.get_data_loader([subject], self.train_sessions, self.gestures, self.train_trials)
            test_loader = self.get_data_loader([subject], self.test_sessions, self.gestures, self.test_trials)

            results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
                    'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
            save_name_pre = '{}_{}_{}_{}_{}_{}'.format(subject, self.num_epochs, self.batch_size, self.feature_dim, self.temperature, self.k)
            save_dir = "{}/{}/{}".format(self.model_dir, self.task, self.dataset_name)
            
            if self.loocv:
                model = linear.Net(self.num_channels, self.window_size, self.num_gestures, pretrained_path=f'{save_dir}/{subject}_pretrained_model.pth').to(self.device)
            else:
                model = linear.Net(self.num_channels, self.window_size, self.num_gestures, pretrained_path=f'{save_dir}/pretrained_model.pth').to(self.device)
            
            for param in model.f.parameters():
                param.requires_grad = False

            # flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).to(self.device),))
            # flops, params = clever_format([flops, params])
            # print('# Model Params: {} FLOPs: {}'.format(params, flops))
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
            loss_criterion = torch.nn.CrossEntropyLoss()
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
            if self.args.task == "inter_session":
                test_loader = self.get_data_loader([subject], test_sessions, gestures, test_trials)
            elif self.args.task == "inter_subject":
                test_loader = self.get_data_loader([subject], sessions, gestures, test_trials)

            save_dir = "{}/{}/{}".format(self.args.model_dir, self.args.task, self.args.dataset_name)
            model = linear.Net(num_channels, window_size, num_gestures, pretrained_path=f'{save_dir}/{subject}_linear_model.pth').to(self.device)
            for param in model.f.parameters():
                param.requires_grad = False

            # flops, params = profile(model, inputs=(torch.randn(1, 1, self.args.window_size, self.args.num_channels).to(self.device),))
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