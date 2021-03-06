import os
from sys import path_hooks
import time
import torch
import numpy as np
import random
from torch.nn.modules import linear
from torch.utils.tensorboard import SummaryWriter

from solver.utils import log, data_reader
from solver.model import stcn
from solver.runner import inter_subject, inter_session

class Solver():
    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args
        # self.init_seed()
        self.init_device()
        self.check_dirs()
        self.get_logger_writer()
        self.get_model()
        self.dump_info()

    def dump_info(self):
        for k, v in vars(self.args).items():
            self.logger.info(f"{k}: {v}")

    def init_device(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def init_seed(self):
        torch.cuda.manual_seed_all(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def check_dirs(self):
        if not os.path.exists(f"{self.args.log_dir}/{self.args.task}/{self.args.dataset_name}"):
            os.makedirs(f"{self.args.log_dir}/{self.args.task}/{self.args.dataset_name}")
        if not os.path.exists(self.args.tb_dir):
            os.makedirs(self.args.tb_dir)
        if not os.path.exists(f"{self.args.model_dir}/{self.args.task}/{self.args.dataset_name}"):
            os.makedirs(f"{self.args.model_dir}/{self.args.task}/{self.args.dataset_name}")

    def get_logger_writer(self):
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        file_name = self.args.dataset_name + f"{t}"
        log_path = os.path.join(self.args.log_dir, self.args.task, self.args.dataset_name)
        self.logger = log.get_logger(log_path, f"{t}.log")
        self.writer = SummaryWriter(os.path.join(self.args.tb_dir, file_name))

    def get_model(self):
        num_gestures= len(self.args.gestures)
        model = stcn.STCN(num_channels=1, num_points=self.args.num_channels, num_classes=num_gestures)
        return model.to(self.device)

    def start(self):
        self.logger.info('==============================================================')
        self.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # self.logger.info(f"Task {self.args.task} {self.args.stage} ")
        self.logger.info(f'Dataset_name: {self.args.dataset_name}')
        self.logger.info(f'Leave one out cross validation: {self.args.loocv}')
        self.logger.info(f'Task: {self.args.task}')
        self.logger.info(f'Stage: {self.args.stage}')
        
        if self.args.task == "intra_session":
            self.intra_session()
        elif self.args.task == "inter_session":
            self.inter_session()
        elif self.args.task == "inter_subject":
            self.inter_subject()
        else:
            raise ValueError

        self.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def intra_session(self):
        subjects = self.args.subjects
        sessions = self.args.sessions
        gestures = self.args.gestures
        trials = self.args.trials

        if self.args.stage == "pretrain":
            pass
        elif self.args.stage == "train":
            accuracy = np.zeros((len(subjects), len(sessions)))

            for i, subject in enumerate(subjects):
                for j, session in enumerate(sessions):
                    self.logger.info(f"Begin training subject {subject} session {session}")

                    self.get_model()
                    path = os.path.join(self.args.model_path, f"pretrain-{self.args.dataset_name}.pkl")
                    if self.args.need_pretrain:
                        if os.path.exists(path):
                            self.model.load_state_dict(torch.load(path))
                            self.logger.info("load pretrain model successfully")
                    train_trials = self.args.train_trials
                    test_trials = self.args.test_trials

                    self.logger.info(f"{subject}, {session}, {gestures}, {train_trials}, {test_trials}")

                    self.train_loader = self.get_loader([subject], [session], gestures, train_trials)
                    self.test_loader = self.get_loader([subject], [session], gestures, test_trials)
                    trial_acc = self.train(subject, session)

                    accuracy[i][j] = trial_acc


            self.logger.info(f"All session accuracy:\n {accuracy}")
            self.logger.info(f"All subject average accuracy:\n {accuracy.mean()}")


        elif self.args.stage == "test":
            accuracy = np.zeros((len(subjects), len(sessions)))

            for i, subject in enumerate(subjects):
                for j, session in enumerate(sessions):
                    self.get_model()
                    path = os.path.join(self.args.model_path, f"trained-{self.args.dataset_name}-subject{subject}-session{session}.pkl")
                    if os.path.exists(path):
                        self.model.load_state_dict(torch.load(path))
                        test_trials = self.args.test_trials
                        self.test_loader = self.get_loader([subject], [session], gestures, test_trials)
                        self.get_optimizer()
                        metric = self.test()
                        test_last_acc = metric["accuracy"]
                        self.logger.info(f"Test subject {subject} session {session}: {test_last_acc}")
                        accuracy[i][j] = test_last_acc
                    else:
                        self.logger.info(f"{path} not exists")
                        exit(1)
            self.logger.info(f"All session accuracy:\n{accuracy}\n Avg: {accuracy.mean()}")

        else:
            raise ValueError

    def inter_session(self):
        if self.args.stage == "pretrain":
            pretrainer = inter_session.Pretrainer(self.args, self.logger)
            pretrainer.start()
        elif self.args.stage == "train":
            trainer = inter_session.Trainer(self.args, self.logger)
            trainer.start()
        elif self.args.stage == "test":
            trainer = inter_session.Trainer(self.args, self.logger)
            trainer.test()
        else:
            raise ValueError

    def inter_subject(self):
        if self.args.stage == "pretrain":
            pretrainer = inter_subject.Pretrainer(self.args, self.logger)
            pretrainer.start()
        elif self.args.stage == "train":
            trainer = inter_subject.Trainer(self.args, self.logger)
            trainer.start()
        elif self.args.stage == "test":
            trainer = inter_subject.Trainer(self.args, self.logger)
            trainer.test()
        else:
            raise ValueError


    def get_loader(self, subjects, sessions, gestures, trials):
        dataset = data_reader.DataReader(subjects, sessions, gestures, trials, self.args)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=0)
        return data_loader


    def train(self, subject, session):
        self.logger.info(f"Begin train")
        self.get_optimizer()

        path = os.path.join(self.args.model_path, f"trained-{self.args.dataset_name}-subject{subject}-session{session}.pkl")
        
        metric = self.test()
        init_acc = metric["accuracy"]

        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            metric = self.test()
            last_acc = metric["accuracy"]
        else:
            last_acc = 0

        self.logger.info("Initial: {}, last {}".format(init_acc, last_acc))

        best_acc = last_acc
        for epoch in range(1, self.args.num_epochs+1):
            self.model.train()
            epoch_loss = 0
            correct = 0

            true_label = []
            pred_label = []
            self.train_loader.dataset.shuffle()
            for step, (x, y) in enumerate(self.train_loader):
                x = x.float().to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                correct += (output.argmax(dim=1) == y).sum()

                true_label.extend(y.tolist())
                pred_label.extend(output.argmax(dim=1).tolist())

            self.scheduler.step()

            train_acc = 1.0 * correct / len(self.train_loader.dataset)


            metric = self.test()
            if True or epoch == self.args.num_epochs:
                # self.draw_confusion(true_label, pred_label, "train")
                # self.draw_confusion(metric["true_label"], metric["pred_label"], "test")
                pass

            if metric["accuracy"] > best_acc:
                best_acc = metric["accuracy"]
                torch.save(self.model.state_dict(), path)


            self.writer.add_scalars(main_tag="loss", tag_scalar_dict={"train_loss": epoch_loss, "valid_loss": metric["loss"]},
                               global_step=epoch)
            self.writer.add_scalars(main_tag="accuracy", tag_scalar_dict={"train_accuracy": train_acc, "valid_acc": metric["accuracy"]},
                               global_step= epoch)

            self.logger.info("Epoch [{:5d}/{:5d}]\t train_loss: {:.08f}\t test_loss: {:.08f}\t\t train_accuracy: {:.06f} [{}/{}]\t test_accuracy: {:.06f} [{}/{}]"
                        .format(epoch, self.args.num_epochs, epoch_loss, metric["loss"],
                                train_acc, correct, len(self.train_loader.dataset),
                                metric["accuracy"], metric["correct"], metric["all"]))

        self.logger.info(f"Best: {best_acc}")
        return best_acc

    def test(self):
        self.model.eval()
        metric = {}
        loss = 0
        correct = 0
        true_label = []
        pred_label = []
        for (x, y) in self.test_loader:
            x = x.float().to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss1 = self.criterion(output, y)
            loss += loss1.item()
            correct += (output.argmax(dim=1) == y).sum()

            true_label.extend(y.tolist())
            pred_label.extend(output.argmax(dim=1).tolist())
            del loss1
            del output

        metric["correct"] = correct
        metric["all"] = len(self.test_loader.dataset)
        metric["accuracy"] = correct*1.0 / len(self.test_loader.dataset)
        metric["loss"] = loss
        metric["pred_label"] = pred_label
        metric["true_label"] = true_label
        return metric

    def draw_confusion(self, true_label, pred_label, title):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import recall_score
        import matplotlib.pyplot as plt

        classes = list(set(true_label))
        classes.sort()
        confusion = confusion_matrix(true_label, pred_label)
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                print("{:3d}".format(confusion[first_index][second_index]), end=" ")
            print("\n")


        # print(len(true_label))
        # print(len(pred_label))
        # plt.figure()
        # plt.imshow(confusion, cmap=plt.cm.Blues)
        # plt.title(title)
        # indices = range(len(classes))
        # plt.xticks(indices, classes)
        # plt.yticks(indices, classes)
        # plt.colorbar()
        # plt.xlabel('pred_label')
        # plt.ylabel('true_label')
        # for first_index in range(len(confusion)):
        #     for second_index in range(len(confusion[first_index])):
        #         plt.text(first_index, second_index, confusion[first_index][second_index])
        # plt.show()

    def get_optimizer(self):
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)


