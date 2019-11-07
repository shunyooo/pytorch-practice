from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from time import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import os

import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.utils.data #データセット読み込み関連 p
import torchvision #画像関連
from torchvision import datasets, models, transforms #画像用データセット諸々

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cloudpickle

# --------- tensorboard log file utils ---------

def extract_scalar_from_event(path):
    event_acc = EventAccumulator(path, size_guidance={'scalars': 0})
    event_acc.Reload() # ログファイルのサイズによっては非常に時間がかかる
    target_key = ['step', 'value', 'wall_time']
    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        scalars[tag] = [{k: event.__getattribute__(k) for k in target_key} for event in events]
    return scalars

def extract_values(dict_list, key):
    return [_dict[key] for _dict in dict_list]

def plot_from_dict(scalar_dict, key, ax=None):
    if ax is None:
        ax = plt

    _s_dict_list = scalar_dict[key]
    x = extract_values(_s_dict_list, 'step')
    y = extract_values(_s_dict_list, 'value')
    ax.plot(x, y, label=key)
    
    ax.legend()
    
    
# --------- pytorch utils ---------

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

def init_model(Model, *args, **kwargs):
    return try_gpu(Model(*args, **kwargs))

# data
def make_dataset_loader(batch_size, download_func):
    transform = transforms.Compose([transforms.ToTensor()])
    def load(train):
        dataset = download_func(root='./data', train=train, download=True, transform=transform)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                     shuffle=True, num_workers=2)
        return dataset_loader
    train_loader, test_loader = load(True), load(False)
    print(f'loaded {download_func} train:{len(train_loader.dataset)}, test:{len(test_loader.dataset)}')
    return train_loader, test_loader

# plot image
def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def to_img_single(x):
    return to_img(x).view(28, 28)

def plot_gray_img(img_tensor, ax=None):
    if ax is None:
        ax = plt
    ax.imshow(img_tensor, cmap = "gray")

def plot_img_grid(imgs, nrow, save_path=None, title=None, show=True):
    # n*nの画像をまとめてplot
    assert nrow**2 == imgs.size(0), f'nrow:{nrow}, nrow**2:{nrow**2} must be == img_size:{imgs.size(0)}'
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, normalize=True)[0]
    if show:
        plt.figure(figsize=(6, 6))
        if title is not None: plt.title(title)
        plot_gray_img(imgs)
    if show and save_path is not None:
        plt.savefig(save_path, dpi=130)
    elif not show and save_path is not None:
        plt.imsave(save_path, imgs, cmap='gray', dpi=130)


# 共通の train test 系
def train(model, optimizer, dataset_loader, total_step, writer):
    model.train()
    train_loss = 0
    for i, data in enumerate(dataset_loader):
        total_step += 1
        
        # optimize
        optimizer.zero_grad()
        loss = model.forward_calculate_loss(data)
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        
        # log
        writer.add_scalar('train/loss', loss.data.item()/data[0].size(0), total_step)
    
    train_loss /= len(dataset_loader.dataset)
    return train_loss, total_step

def test(model, dataset_loader, epoch, total_step, writer):
    model.eval()
    test_loss = 0
    for i, data in enumerate(dataset_loader):
        loss = model.forward_calculate_loss(data)
        test_loss += loss.data.item()
    test_loss /= len(dataset_loader.dataset)
    # log
    writer.add_scalar('test/loss', test_loss, total_step)
    return test_loss

def get_now_str():
    now = datetime.datetime.now()
    return now.strftime('%Y%m%dT%H%M%S')

def save_model(model, path):
    with open(path, 'wb') as f:
        cloudpickle.dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        return cloudpickle.load(f)
        
def run_train_test(model, model_name, task_name, optimizer, num_epochs, train_loader, test_loader):
    timer = start_str_timer(num_epochs)
    # log path
    log_dir = f'./runs/{task_name}/{model_name}/{get_now_str()}'
    model_save_dir = f'./models/{task_name}/{model_name}/{get_now_str()}'
    os.makedirs(log_dir)
    os.makedirs(model_save_dir)
    print(f'make dir {log_dir}, {model_save_dir}')
    
    # train - valid
    writer = SummaryWriter(log_dir=log_dir)
    total_step = 0
    for epoch in range(1, num_epochs+1):
        train_loss, total_step = train(model, optimizer, train_loader, total_step, writer)
        test_loss = test(model, test_loader, epoch, total_step, writer)
        print(f'epoch [{epoch}/{num_epochs}], step:{total_step}, loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, {timer(epoch)}')
        model_save_path = f'{model_save_dir}/epoch{epoch:03}.pt'
        print(f'model save to {model_save_path}')
        save_model(model, model_save_path)
        
    return {'model':model, 'log_dir':log_dir, 'model_dir':model_save_dir}
    

# --------- log utils ---------

def get_d_h_m_s(sec):
    td = datetime.timedelta(seconds=sec)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    v_list = [td.days, h, m, s]
    unit_list = ['日', '時間', '分', '秒']
    # to_text
    tt = lambda v, unit: f'{v}{unit}' if v > 0 else None
    text = ''.join([tt(v, unit)
                  for v, unit in zip(v_list, unit_list) if v > 0])
    return text

def start_str_timer(num_goal=None):
    # 経過時間を読みやすい形で返す.
    # num_goalが与えられた場合, 終了予測時間も返却する.
    start = time()
    def elapsed(num_step=None):
        elapsed_sec = time()-start
        res = {'elapsed': get_d_h_m_s(elapsed_sec)}
        if num_goal is not None and num_step is not None and num_step != 0:
            # 残りステップ数 * 平均時間毎ステップ
            remain_sec = (num_goal-num_step)*(elapsed_sec/num_step)
            res['remain'] = get_d_h_m_s(remain_sec)
        return res
    return elapsed



