import time
import os as os
import numpy as np
import torch
import torch.nn as nn
from IPython import embed

import random
from torch.autograd import Variable
from .checkpoints.save_load import save_checkpoint

# from util.transforms import mixup_data, mixup_criterion
# from checkpoints import save_checkpoint

class Trainer(object):
    def __init__(self, loss_computation, model, criterion, budget, optimizer, scheduler, budget_type, device, images_to_plot=0, checkpoint_path=None, config_id=None, pred_save_dir=None):
        self.checkpoint_path = checkpoint_path
        self.config_id = config_id

        self.scheduler = scheduler
        # if self.scheduler and not hasattr(self.scheduler, 'cumulative_time'):
        #     self.scheduler.cumulative_time = 0
        self.optimizer = optimizer
        self.device = device

        self.budget = budget
        self.loss_computation = loss_computation

        self.images_plot_count = images_to_plot

        self.budget_type = budget_type
        self.cumulative_time = 0

        self.train_loss_sum = 0
        self.train_iterations = 0

        self.latest_checkpoint = None

        self.pred_save_dir = os.path.join(pred_save_dir, "predictions")
        
        if self.pred_save_dir is not None:
            os.makedirs(self.pred_save_dir, exist_ok=True)

        try:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            self.model = model.to(self.device)
        except:
            print("CUDA unavailable, continue using CPU.")
            self.model = model.to("cpu")

        #try:
        self.criterion = criterion.to(self.device)
        #except:
        #    print("No criterion specified.")
        #    self.criterion = None

    def train(self, epoch, train_loader, metrics):
        '''
            Trains the model for a single epoch
        '''

        # train_size = int(0.9 * len(train_loader.dataset.train_data) / self.config.batch_size)
        loss_sum = 0.0
        N = 0

        # print('\33[1m==> Training epoch # {}\033[0m'.format(str(epoch)))


        classified = []
        misclassified = []

        preds = []

        self.model.train()

        budget_exceeded = False
        metric_results = [0] * len(metrics)
        start_time = time.time()
        for step, (data, targets) in enumerate(train_loader):
            # import matplotlib.pyplot as plt
            # img = plt.imshow(data.numpy()[0,1,:])
            # plt.show()

            # images += list(data.numpy())
            # print('Data:', data.size(), ' - Label:', targets.size())

            data = data.to(self.device)
            targets = targets.to(self.device)

            data, criterion_kwargs = self.loss_computation.prepare_data(data, targets)
            batch_size = data.size(0)

            outputs = self.model(data)
            loss_func = self.loss_computation.criterion(**criterion_kwargs)
            loss = loss_func(self.criterion, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #embed()

            # print('Train:', ' '.join(str(outputs).split('\n')[0:2]))

            if self.images_plot_count > 0:
                with torch.no_grad():
                    _, pred = outputs.topk(1, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(targets.view(1, -1).expand_as(pred)).cpu().numpy()[0]
                    data = data.cpu().numpy()
                    classified += list(data[correct.astype(bool)])
                    misclassified += list(data[(1-correct).astype(bool)])
                    if len(classified) > self.images_plot_count:
                        classified = random.sample(classified, self.images_plot_count)
                    if len(misclassified) > self.images_plot_count:
                        misclassified = random.sample(misclassified, self.images_plot_count)

            # self.scheduler.cumulative_time += delta_time
            # self.scheduler.last_step = self.scheduler.cumulative_time - delta_time - 1e-10

            tmp = time.time()

            #for i, metric in enumerate(metrics):
            #    log['train_' + metric.name] = train_metrics_results[i]

            #    if self.val_loader is not None:
            #        log['val_' + metric.name] = valid_metric_results[i]
            
            with torch.no_grad():
                for i, metric in enumerate(metrics):
                    metric_results[i] += self.loss_computation.evaluate(metric, outputs, **criterion_kwargs) * batch_size

            if self.pred_save_dir is not None:
                preds.append(outputs.detach().cpu().tolist())

            loss_sum += loss.item() * batch_size
            N += batch_size

            #print('Update', (metric_results[0] / N), 'loss', (loss_sum / N), 'lr', self.optimizer.param_groups[0]['lr'])

            if self.budget_type == 'time' and self.cumulative_time + (time.time() - start_time) >= self.budget:
                # print(' * Stopping at Epoch: [%d][%d/%d] for a budget of %.3f s' % (epoch, step + 1, train_size, self.config.budget))
                budget_exceeded = True
                break

        if N==0: # Fixes a bug during initialization
            N=1

        if self.images_plot_count > 0:
            import tensorboard_logger as tl
            tl.log_images('Train_Classified/Image', classified, step=epoch)
            tl.log_images('Train_Misclassified/Image', misclassified, step=epoch)

        if self.checkpoint_path and self.scheduler.snapshot_before_restart and self.scheduler.needs_checkpoint():
            self.latest_checkpoint = save_checkpoint(self.checkpoint_path, self.config_id, self.budget, self.model, self.optimizer, self.scheduler)

        try:
            self.scheduler.step(epoch=epoch)
        except:
            self.scheduler.step(metrics=loss_sum / N, epoch=epoch)

        self.cumulative_time += (time.time() - start_time)
        #print('LR', self.optimizer.param_groups[0]['lr'], 'Update', (metric_results[0] / N), 'loss', (loss_sum / N))

        if self.pred_save_dir is not None:
            np.save(os.path.join(self.pred_save_dir, "outputs_ep_"+str(epoch)), np.array(preds))

        return [res / N for res in metric_results], loss_sum / N, budget_exceeded


    def evaluate(self, test_loader, metrics, epoch=0):

        N = 0
        metric_results = [0] * len(metrics)
        
        classified = []
        misclassified = []

        self.model.eval()

        with torch.no_grad():
            for step, (data, targets) in enumerate(test_loader):

                # import matplotlib.pyplot as plt
                # img = plt.imshow(data.numpy()[0,1,:])
                # plt.show()

                try:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                except:
                    data = data.to("cpu")
                    targets = targets.to("cpu")

                batch_size = data.size(0)

                outputs = self.model(data)

                if self.images_plot_count > 0:
                    _, pred = outputs.topk(1, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(targets.view(1, -1).expand_as(pred)).cpu().numpy()[0]
                    data = data.cpu().numpy()
                    classified += list(data[correct.astype(bool)])
                    misclassified += list(data[(1-correct).astype(bool)])
                    if len(classified) > self.images_plot_count:
                        classified = random.sample(classified, self.images_plot_count)
                    if len(misclassified) > self.images_plot_count:
                        misclassified = random.sample(misclassified, self.images_plot_count)

                # print('Valid:', ' '.join(str(outputs).split('\n')[0:2]))
                # print('Shape:', outputs.shape, 'Sums', str(outputs.cpu().numpy().sum(1)).replace('\n', ''))
                
                for i, metric in enumerate(metrics):
                    metric_results[i] += metric(outputs.data, targets.data) * batch_size

                N += batch_size

        if self.images_plot_count > 0:
            import tensorboard_logger as tl
            tl.log_images('Valid_Classified/Image', classified, step=epoch)
            tl.log_images('Valid_Misclassified/Image', misclassified, step=epoch)

        self.model.train()
            
        return [res / N for res in metric_results]
    

    def class_to_probability_mapping(self, test_loader):

        N = 0

        import numpy as np
        import torch.nn as nn
        
        probs = None;
        class_to_index = dict()
        target_count = []
        
        self.model.eval()

        with torch.no_grad():
            for i, (data, targets) in enumerate(test_loader):
    
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                batch_size = data.size(0)

                outputs = self.model(data)

                for i, output in enumerate(outputs):
                    target = targets[i].cpu().item()
                    np_output = output.cpu().numpy()
                    if target not in class_to_index:
                        if probs is None:
                            probs = np.array([np_output])
                        else:
                            probs = np.vstack((probs, np_output))
                        class_to_index[target] = probs.shape[0] - 1
                        target_count.append(0)
                    else:
                        probs[class_to_index[target]] = probs[class_to_index[target]] + np_output

                    target_count[class_to_index[target]] += 1

                N += batch_size
            
            probs = probs / np.array(target_count)[:, None] #np.max(probs, axis=1)[:, None]
            probs = torch.from_numpy(probs)
            # probs = nn.Softmax(1)(probs)

        self.model.train()
        return probs, class_to_index
