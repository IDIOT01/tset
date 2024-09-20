import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from einops import rearrange
from collections import OrderedDict
from functools import partial
from typing import Callable
import torch.optim as optim
from tqdm import tqdm
# from utility import count_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from einops import rearrange
from collections import OrderedDict
from functools import partial
from typing import Callable
import torch.optim as optim
from tqdm import tqdm
# from utility import count_parameters


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


# 定义cifar-10的CNN卷积层
class CifarConv(nn.Module):
    def __init__(self):
        super(CifarConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First conv layer
        x = self.conv1(x)  # Convolution
        x = self.bn1(x)  # Batch Normalization
        x = self.relu(x)  # ReLU Activation
        x = self.pool(x)  # Max Pooling

        # Second conv layer
        x = self.conv2(x)  # Convolution
        x = self.bn2(x)  # Batch Normalization
        x = self.relu(x)  # ReLU Activation
        x = self.pool(x)  # Max Pooling

        # Third conv layer
        x = self.conv3(x)  # Convolution
        x = self.bn3(x)  # Batch Normalization
        x = self.relu(x)  # ReLU Activation
        x = self.pool(x)  # Max Pooling

        # Fourth conv layer
        x = self.conv4(x)  # Convolution
        x = self.bn4(x)  # Batch Normalization
        x = self.relu(x)  # ReLU Activation
        x = self.pool(x)  # Max Pooling

        # x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)
        return x


# 定义CNN的MLP层
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim // 2, class_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, class_num, mlp_hidden_dim):
        super(CNN, self).__init__()
        self.conv = CifarConv()
        self.mlp = MLP(512 * 2 * 2, mlp_hidden_dim, class_num)

    def forward(self, x):
        aux_loss = None
        return self.mlp(self.conv(x)), aux_loss


# 定义Gating网络
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, expert_num):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, expert_num))

    def forward(self, x):
        return self.fc(x)


class CNN_moe_noise(nn.Module):
    def __init__(self, class_num, expert_num, top_k, mlp_hidden_dim, noisy_gating=True, gating_hidden_dim=2048,
                 gating_drop_out=0.5):
        super(CNN_moe_noise, self).__init__()
        self.conv = CifarConv()
        self.experts = nn.ModuleList(MLP(512 * 2 * 2, mlp_hidden_dim, class_num) for _ in range(expert_num))
        self.k = top_k
        self.class_num = class_num
        self.num_experts = expert_num
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.noisy_gating = noisy_gating
        self.w_gate = nn.Sequential(
            nn.Linear(512 * 2 * 2, gating_hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(gating_hidden_dim, expert_num)
        )
        self.w_noise = nn.Sequential(
            nn.Linear(512 * 2 * 2, gating_hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(gating_hidden_dim, expert_num)
        )
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= expert_num)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-3):
        x = self.conv(x)
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss

# if __name__ == '__main__':
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ])

#     train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#     model = CNN_moe_noise(class_num=10,expert_num=4, top_k=1, mlp_hidden_dim=2048)
#     model_cnn = CNN(class_num=10,mlp_hidden_dim=2048)
#     print('*'*80)
#     a, b = count_parameters(model)
#     a1, b1 = count_parameters(model_cnn)
#     print('CNN_moe parameters number is {} and storage is {}'.format(a,b))
#     print('CNN parameters number is {} and storage is {}'.format(a1,b1))
#     print('*'*80)
#     # model = CNN(class_num=10,mlp_hidden_dim=256)
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     # Training and Evaluation
#     num_epochs = 30
#     for epoch in range(num_epochs):
#         model.noisy_gating = True
#         model.train()
#         running_loss = 0.0
#         gating_total_loss = 0.0
#         progress_bar = tqdm(train_loader, desc="Training", leave=False)
#         for images, labels in progress_bar:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs, gating_loss = model(images)
#             loss = criterion(outputs, labels)+ gating_loss
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()*images.size(0)
#             gating_total_loss += gating_loss.item()*images.size(0)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_gating_loss = gating_total_loss / len(train_loader.dataset)
#         print('*'*100)
#         print(epoch_gating_loss)
#         train_loss = epoch_loss
#         model.noisy_gating = False
#         model.eval()
#         correct = 0
#         total = 0
#         running_loss = 0.0
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs, gating_loss = model(images)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item() * images.size(0)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         epoch_loss = running_loss / len(test_loader.dataset)
#         accuracy = correct / total
#         test_loss = epoch_loss
#         test_accuracy = accuracy
#         print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
#         print('*'*100)
