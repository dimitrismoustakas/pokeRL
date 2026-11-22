"""Sample learner decisions using stable inputs for one fixed-policy rollout."""

import torch

from ..env.actions import NUM_TERA_CHOICES
from .inference_batch import InputTransfer


class RolloutPolicy:
    def __init__(self, agent, obs, masks, device, tera_exploration_prob):
        self.agent = agent
        self.device = device
        self.tera_exploration_prob = tera_exploration_prob
        self.observations = InputTransfer(obs, device)
        self.masks = InputTransfer(masks, device, force_bool=True)
        self.graph = None

    def _forward(self, hidden, **kwargs):
        return self.agent(
            self.observations.target.tensors,
            hidden,
            action_masks=self.masks.target.tensors,
            tera_exploration_prob=self.tera_exploration_prob,
            **kwargs,
        )

    def _capture(self, hidden):
        self.hidden = (
            None if hidden is None else tuple(state.clone() for state in hidden)
        )
        batch_size = self.observations.target.capacity
        actions = self.agent.action_head.num_actions
        self.draws = tuple(
            torch.ones(batch_size, size, device=self.device)
            for size in (actions, actions, NUM_TERA_CHOICES)
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward(self.hidden, sampling_exponentials=self.draws)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self._forward(self.hidden, sampling_exponentials=self.draws)

    @torch.no_grad()
    def sample(self, hidden):
        if self.device.type != "cuda":
            return self._forward(hidden)
        if self.graph is None:
            self._capture(hidden)
        if hidden is not None:
            for dest, state in zip(self.hidden, hidden, strict=True):
                dest.copy_(state)
        # Draw the two move vectors and the Tera vector in eager-policy order.
        for draw in self.draws:
            draw.exponential_()
        self.graph.replay()
        return self.output
