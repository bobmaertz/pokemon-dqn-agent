import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple

from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import ListStorage

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class DeepQLearningAgent:
    """
    Deep Q-Learning agent for Pokémon Blue
    """

    def __init__(
            self,
            state_size,
            action_size,
            replay_memory_size=500,
            replay_warmup=None,
            minibatch_size=64,
            epsilon_decay=0.99,
            learning_rate=0.01,
            epsilon_min=0.01,
            epsilon_start=1.0,
            gamma=0.95,
            target_update_every=250,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory_size = replay_memory_size
        # TorchRL replay buffer with a Python-object storage.
        # We use an identity collate_fn so sampling returns a plain list of
        # Transition-like objects (rather than attempting to stack).
        self.replay_memory = ReplayBuffer(
            storage=ListStorage(max_size=self.replay_memory_size),
            sampler=RandomSampler(),
            collate_fn=lambda x: x,
        )
        self.gamma = float(gamma)    # discount rate
        self.epsilon = float(epsilon_start)   # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self._train_counter = 0
        self.minibatch_size = minibatch_size

        if replay_warmup is None:
            # Default warmup is small enough to start learning early, but still
            # allow a bit of experience collection. Also capped by replay size.
            replay_warmup = min(int(self.replay_memory_size), 1000)
        self.replay_warmup = int(replay_warmup)
        self.target_update_every = int(target_update_every)

        self.device = self._select_device()

        # Neural Network for Q-learning
        self.policy_model = self._build_model()
        self.policy_model.to(self.device)

        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.to(self.device)

        self.optimizer = optim.AdamW(
            self.policy_model.parameters(),
            lr=self.learning_rate,
            amsgrad=True)

    def _select_device(self):
        # Keep tests deterministic and avoid requiring CUDA/MPS builds.
        if os.getenv("PYTEST_CURRENT_TEST") is not None:
            return torch.device("cpu")

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")

        if getattr(torch.backends, "cuda", None) is not None and torch.backends.cuda.is_built() and torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _state_to_tensor(self, state, *, add_batch_dim: bool):
        state_arr = np.asarray(state, dtype=np.float32)

        # Expected inputs:
        # - (H, W) uint8 grayscale
        # - (1, H, W) channel-first
        # - (N, H, W) batches in training (handled separately)
        if state_arr.ndim == 2:
            state_arr = state_arr[None, ...]  # (1, H, W)

        if state_arr.ndim != 3:
            raise ValueError(f"Expected state with 2 or 3 dims, got shape {state_arr.shape}")

        if add_batch_dim:
            state_arr = state_arr[None, ...]  # (1, 1, H, W)

        # Normalize if looks like uint8 images.
        if state_arr.max(initial=0.0) > 1.0:
            state_arr = state_arr / 255.0

        return torch.from_numpy(state_arr).to(device=self.device)

    def _build_model(self):
        """
        Create Deep Neural Network for Q-Learning
        """
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 36 * 40, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        )

        return model

    def act(self, state):
        """
        Choose action using epsilon-greedy strategy
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = self._state_to_tensor(state, add_batch_dim=True)
        q_values = self.policy_model(state_tensor)
        return torch.argmax(q_values).item()

    def update_memory(self, transition):
        """
        Update the replay memory with the latest transition tuple
        - state, action, reward, next_state, done
        """
        self.replay_memory.add(transition)

    def _sample_minibatch(self):
        return self.replay_memory.sample(self.minibatch_size)

    def update_target_network(self):
        """
        Update the target network with the policy network weights
        """
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def train(self):
        """
        Train the agent using Deep Q-Learning;

        returns:
         - Loss for training
        """
        # Don't train until we have enough transitions to (a) meet the warmup
        # threshold and (b) sample a full minibatch.
        min_required = max(self.replay_warmup, self.minibatch_size)
        if len(self.replay_memory) < int(min_required):
            return

        minibatch = self._sample_minibatch()

        # Reviewing algorithm from https://www.youtube.com/watch?v=qfovbG84EBg&t=335s
        # TODO: Double check normalization of 255
        current_states_np = np.asarray([transition.state for transition in minibatch], dtype=np.float32)
        if current_states_np.ndim == 3:
            current_states_np = current_states_np[:, None, :, :]  # (B, 1, H, W)
        current_states_np /= 255.0
        current_states = torch.from_numpy(current_states_np).to(self.device)

        actions_np = np.fromiter((t.action for t in minibatch), dtype=np.int64, count=self.minibatch_size)
        rewards_np = np.fromiter((t.reward for t in minibatch), dtype=np.float32, count=self.minibatch_size)
        dones_np = np.fromiter((t.done for t in minibatch), dtype=np.float32, count=self.minibatch_size)

        actions = torch.from_numpy(actions_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        next_states_np = np.asarray([transition.next_state for transition in minibatch], dtype=np.float32)
        if next_states_np.ndim == 3:
            next_states_np = next_states_np[:, None, :, :]  # (B, 1, H, W)
        next_states_np /= 255.0
        next_states = torch.from_numpy(next_states_np).to(self.device)

        # Compute Q-values for current states
        curr_q = self.policy_model(current_states)
        curr_q = curr_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q-values for next states using target network
        next_q = self.target_model(next_states).detach()
        max_next_q = next_q.max(1)[0]

        # Compute target Q-values
        target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(curr_q, target_q)
        loss_value = loss.item()
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_counter += 1

        # Update target network every N training steps (N, 2N, ...)
        if self.target_update_every > 0 and (self._train_counter % self.target_update_every == 0):
            self.update_target_network()
        return loss_value, curr_q.mean().item()

    def save(self, name):
        """
        Save model weights
        """
        # Save the model's state_dict
        model_filename = f"{name}_model_state_dict.pth"
        torch.save(self.target_model.state_dict(), model_filename)

        optimizer_filename = f"{name}_optimizer_state_dict.pth"
        torch.save(self.optimizer.state_dict(), optimizer_filename)

