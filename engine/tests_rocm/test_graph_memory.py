"""Regression for live BLAS workspaces growing on serial HTTP recaptures."""

import gc
import tempfile
import threading
import unittest

import torch
from fixture import make_checkpoint

from rec_rocm.model import FlintModel, GraphDecoder


class GraphMemory(unittest.TestCase):
    def test_context_switches_from_new_threads_have_bounded_live_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FlintModel(make_checkpoint(directory), capacity=1024)
            model.rdna4_graph = "compact"
            allocations, errors = [], []

            @torch.inference_mode()
            def request(index):
                try:
                    model.reset()
                    model.decode_attention_capacity = 512 if index % 2 else 1024
                    model.forward(torch.arange(16, device=model.device))
                    saved = model.snapshot()
                    token = torch.tensor([17], device=model.device)
                    expected = model.forward(token).clone()
                    model.restore(saved)
                    graph = GraphDecoder(model)
                    torch.testing.assert_close(graph(token), expected, atol=0, rtol=0)
                    del graph, expected, token, saved
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    allocations.append(torch.cuda.memory_allocated())
                except BaseException as error:
                    errors.append(error)

            # New request threads exercise BLAS handle reuse as well as the
            # changing Graph shape. Exclude initial handle/shape warmup only.
            for index in range(12):
                thread = threading.Thread(target=request, args=(index,))
                thread.start()
                thread.join()
                if errors:
                    raise errors[0]
            self.assertLessEqual(max(allocations[3:]) - min(allocations[3:]), 16 * 1024**2)
