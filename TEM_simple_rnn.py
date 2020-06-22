import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from config import cfg
import time
import os

#from fast_weights import fast_weights_model
from generate import Graph4D
from torch.distributions.kl import kl_divergence
import scores
from tensorboardX import SummaryWriter
from util import Checkpointer

import util

SEED = 9101
torch.manual_seed(SEED)
np.random.seed(SEED)

def softmax_cross_entropy_with_logits(logits, labels):
    loss = torch.sum(-labels * F.log_softmax(logits, -1), -1)
    return loss

class TEM(nn.Module):
    """docstring for TEM"""
    def __init__(self, batch_size=2, g_dim=45, p_dim=400, x_dim=45, hidden_dim=128, a_dim=4, steps=128, n_s_star=10, device=cfg.device):
        super(TEM, self).__init__()
        self.g_dim = g_dim
        self.p_dim = p_dim
        self.a_dim = a_dim
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.batch_size = batch_size
        self.steps = steps
        self.n_s_star = n_s_star

        #self.hebbian = fast_weights_model(self.batch_size, self.steps, self.g_dim, self.p_dim)

        self.alpha = nn.Parameter(torch.randn(1, dtype=torch.float32, device=cfg.device), requires_grad=True)

        self.w_p = nn.Parameter(torch.randn(1, dtype=torch.float32, device=cfg.device), requires_grad=True)

        # self.p_t_enc = nn.Sequential(
        #     Preprocess_img(),
        #     nn.Conv2d(3, 64, kernel_size=2, stride=2),
        #     nn.LeakyReLU(0.01),
        #     nn.Conv2d(64, 16, kernel_size=2, stride=2),
        #     nn.LeakyReLU(0.01),
        #     Flatten()
        # )

        # Input: Grid cell: g_dim 
        self.place_cell_rnn = nn.RNN(self.g_dim, self.p_dim, self.steps)

        self.sigma_t = Normal(0, torch.tensor(np.identity(self.g_dim)))

        #f_D(): MLP
        self.f_D = nn.Sequential(
            nn.Linear(self.a_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            )

        #f_mu_g(): Linear with Tanh
        self.f_mu_g = nn.Sequential(
            nn.Linear(self.g_dim, self.g_dim),
            nn.Tanh(),
            )

        #f_sigma_g(): MLP
        self.f_sigma_g = nn.Sequential(
            nn.Linear(self.g_dim, 20),
            nn.ReLU(),
            nn.Linear(20, self.g_dim),
            )

        #f_p for inferencing place cells
        self.f_p = nn.Sequential(
            nn.Linear(self.x_dim, self.p_dim),
            nn.LeakyReLU(),
            )

        #f_sigma_place
        self.f_sigma_p = nn.Sequential(
            nn.Linear(self.p_dim, 200),
            nn.ReLU(),
            nn.Linear(200, self.p_dim)
            )

    def forward(self, bx, ba):
        # if self.training_wo_wall:
        #     action_one_hot_value, position, action_selection = random_walk_wo_wall(self)
        # else:
        #     action_one_hot_value, position, action_selection = random_walk(self)

        """
        Number of catergories = 45
        x_t : B * T * X
        a_t : B * T * A
        grid_mu_tensor : T * B * G 
        """

        bx = torch.tensor(bx).float()
        ba = torch.tensor(ba).float()
        #print(ba.shape)

        #Two phases: observation phase and prediction phase 
        grid_mu_list = []
        grid_sigma_list = []
        place_mu_list = []
        place_sigma_list = []
        place_inferred_mu_list = []
        place_inferred_sigma_list = []
        xt_prediction_list = []

        # Generative model: construct grid cells g_t
        for t in range(self.steps):
            if t == 0:
                grid_mu_t = torch.zeros((self.batch_size, self.g_dim), dtype=torch.float32, device=cfg.device)
                #grid_mu_list.append(grid_mu_t)
            else:
                #a_t: B * A
                a_t = ba[:, t - 1, :]
                #print(a_t.type(), a_t.shape)
                grid_with_da = grid_mu_list[t-1] + \
                    self.f_D(a_t) * grid_mu_list[t-1]
                #print(self.sigma_t.sample().type())
                grid_sigma_t = torch.mm(self.f_sigma_g(grid_mu_list[t-1]), self.sigma_t.sample().float())
                grid_mu_t = self.f_mu_g(grid_with_da)
                grid_sigma_list.append(grid_sigma_t)
            grid_mu_list.append(grid_mu_t)

        grid_sigma_t_last = torch.mm(self.f_sigma_g(grid_mu_list[self.steps-1]), self.sigma_t.sample().float())
        grid_sigma_list.append(grid_sigma_t_last)

        grid_mu_tensor = torch.cat(grid_mu_list, 0).view(self.steps, self.batch_size, self.g_dim)
        grid_sigma_tensor = torch.cat(grid_sigma_list, 0).view(self.steps, self.batch_size, self.g_dim)

        # Inference network: construct inferred place cells ~p_t
        x_t_inference = torch.zeros((self.batch_size, self.steps, self.x_dim), device=cfg.device)
        for i in range(self.steps):
            x_t_inference[:, i, :] = (1 - self.alpha) * bx[:, i, :] + self.alpha * bx[:, i, :]
        f_n_x_t = nn.functional.normalize(torch.relu(x_t_inference), p=2, dim=2, eps=1e-12, out=None)
        
        for i in range(self.steps):
            place_inferred_mu_t = torch.bmm(grid_mu_tensor[i, :, :].unsqueeze(1), \
                            self.w_p * f_n_x_t[:, i, :].unsqueeze(1).repeat(1, self.g_dim, 1))
            place_inferred_mu_t = self.f_p(place_inferred_mu_t)
            place_inferred_mu_list.append(place_inferred_mu_t)

        place_inferred_mu_tensor = torch.cat(place_inferred_mu_list, 0).view(self.batch_size, self.steps, self.p_dim)
        place_inferred_sigma_tensor = self.f_sigma_p(place_inferred_mu_tensor)
        #place_inferred_distribution = Normal(place_inferred_mu_tensor, place_inferred_sigma_tensor)


        #Generative model: construct place cells p_t from hebbian memory
        #M_list, place_cell_list, g_t_loss, g_t_acc = self.hebbian(grid_mu_tensor, grid_inferred_mu_tensor)
        
        #place_generated_mu_tensor: B * T * G
        place_generated_mu_tensor, _ = self.place_cell_rnn(grid_mu_tensor.transpose(0,1))
        place_generated_sigma_tensor = self.f_sigma_p(place_generated_mu_tensor)
        #place_generated_distribution = Normal(place_generated_mu_tensor, place_generated_sigma_tensor)

        #Computing loss of ELBO
        loss = nn.MSELoss()(place_generated_mu_tensor, place_inferred_mu_tensor)

        # for t in range(self.steps):
        #     if t == 0:
        #         place_mu_t = torch.zeros((self.batch_size, self.p_dim), devicce=device)
        #         M_t = self.hebbian(grid_observation_list[t], )
        #     index_mask = torch.zeros((self.batch_size, 3, 32, 32), device=device)
        #     for index_sample in range(self.batch_size):
        #         po
        return loss, place_inferred_mu_tensor

def train(save = 0, verbose = 0):
    batch_size = 2
    p_dim = 128
    model = TEM(batch_size=2, g_dim=45, p_dim=128)
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    writer = SummaryWriter(logdir=os.path.join(cfg.logdir, cfg.exp_name), flush_secs=30)
    checkpointer = Checkpointer(os.path.join(cfg.checkpointdir, cfg.exp_name))
    start_epoch = 0
    start_epoch = checkpointer.load(model, optimizer)
    batch_idxs = 20
    graph = Graph4D(num_envs=20, env_size=(8,8), steps=128, num_categories=45)
    observations = graph.observations
    actions = graph.actions
    positions = graph.positions
    positions = (positions - 4) / 8

    # Store the grid scores
    grid_scores = dict()
    grid_scores['btln_60'] = np.zeros((p_dim,))
    grid_scores['btln_90'] = np.zeros((p_dim,))
    grid_scores['btln_60_separation'] = np.zeros((p_dim,))
    grid_scores['btln_90_separation'] = np.zeros((p_dim,))
    grid_scores['lstm_60'] = np.zeros((p_dim,))
    grid_scores['lstm_90'] = np.zeros((p_dim,))
    
    # Create scorer
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, ((-1.1, 1.1), (-1.1, 1.1)),
                                          masks_parameters)

    #place_cells_list = []

    saver_results_directory = 'results/'

    for epoch in range(start_epoch, 20):
        place_cells_list = []
        for idx in range(10):
            gloabl_step = epoch * 20 + idx + 1
            #print(ar_data.train._x)
            bx, ba = observations[idx*batch_size:(idx+1)*batch_size,:,:], actions[idx*batch_size:(idx+1)*batch_size,:,:]
            loss, place_cells = model(bx, ba)
            place_cells_list.append(place_cells)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/kl', loss, gloabl_step)
            #writer.add_scalar('acc/acc', acc, gloabl_step)
            if verbose > 0 and idx % verbose == 0:
                print('Epoch: [{:4d}] [{:4d}/{:4d}] time: {:.4f}, loss: {:.8f}'.format(
                    epoch, 2*idx, batch_idxs, time.time() - start_time, loss
                ))

        #print(len(place_cells_list),place_cells_list[0].shape)
        place_cells_tensor = torch.cat(place_cells_list, 0).detach().numpy()
        print(positions.shape)
        #print(positions)
        filename = 'rates_and_sac_latest_hd_{}.pdf'.format(epoch)
        if epoch % 4 == 0:
            grid_scores['btln_60'], grid_scores['btln_90'], grid_scores['btln_60_separation'], grid_scores['btln_90_separation'] \
                        = util.get_scores_and_plot(latest_epoch_scorer, positions, place_cells_tensor, saver_results_directory, filename)
                        #= util.get_scores_and_plot(latest_epoch_scorer, torch.randn(20,128,2).detach().numpy(), torch.randn(20,128,128).detach().numpy(), saver_results_directory, filename)
    #checkpointer.save(model, optimizer, epoch+1)

if __name__ == "__main__":
    train(verbose = 1)

        