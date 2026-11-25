"""Offline regression checks using a randomly initialized, two-layer grouped-query Llama."""
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from bbh_eval import ctkv


class TinyLlamaCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        config = LlamaConfig(
            vocab_size=45, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        )
        config._attn_implementation = "sdpa"
        self.model = LlamaForCausalLM(config).eval()
        self.instruction = [1, 11, 12]
        self.demos = [
            ([15, 13, 17, 14, 5, 6, 9], 4, 6),
            ([18, 20, 8, 12, 7, 9], 4, 5),
            ([33, 22, 23, 3, 4, 9], 4, 5),
        ]
        self.flat = self.instruction + [token for demo in self.demos for token in demo[0]]
        self.query = [12, 8, 30, 29, 19]

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def fit(self, **kwargs):
        return ctkv.fit(self.model, self.instruction, self.demos, **kwargs)

    def logits(self, prefix, mask=None):
        ids = torch.tensor([self.query])
        count = prefix.layers[0][0].shape[2] + len(self.query)
        mask = torch.ones(1, count, dtype=torch.bool) if mask is None else mask
        out = ctkv._forward(self.model, prefix.layers, prefix.original_length, ids, mask)
        return self.model.lm_head(out.last_hidden_state)

    def test_full_and_masked_reference_logits(self):
        full, _ = self.fit(steps=0)
        direct = self.model(torch.tensor([self.flat + self.query])).logits[:, -len(self.query):]
        torch.testing.assert_close(direct, self.logits(full))
        for method in ("random", "answer_attention"):
            with self.subTest(method=method):
                identity, _ = self.fit(method=method, budget=1, steps=0)
                torch.testing.assert_close(self.logits(identity), direct)
                compact, meta = self.fit(method=method, budget=0.5, steps=0)
                mask = torch.zeros(1, len(self.flat) + len(self.query), dtype=torch.bool)
                mask[:, :len(self.instruction)] = True
                mask[:, len(self.flat):] = True
                for i in meta["retained_demo_indices"]:
                    mask[:, len(self.instruction) + i] = True
                torch.testing.assert_close(self.logits(compact), self.logits(full, mask))

    def test_leave_one_out_blocks_held_out_gradients(self):
        full, _ = self.fit(steps=0)
        params = [param.requires_grad_(True) for layer in full.layers for param in layer]
        owners = torch.tensor([-1]*3 + [0]*7 + [1]*6 + [2]*6)
        ctkv._loss(self.model, full, owners, [0], self.demos, (), torch.Generator().manual_seed(0)).backward()
        for param in params:
            self.assertEqual(param.grad[:, :, 3:10, :].count_nonzero().item(), 0)
            self.assertGreater(param.grad[:, :, :3, :].abs().sum().item(), 0)

    def test_update_and_greedy_cache_parity(self):
        class TokenizerStub:
            eos_token_id = -1

            def decode(self, ids, **kwargs):
                return str(ids)

        full, _ = self.fit(steps=0)
        _, cached = ctkv.decode(self.model, TokenizerStub(), full, self.query, 3)
        direct = []
        for _ in range(3):
            ids = torch.tensor([self.flat + self.query + direct])
            token = self.model(ids).logits[:, -1].argmax(-1).item()
            direct.append(token)
            if token == self.model.config.eos_token_id:
                break
        self.assertEqual(cached, direct)
        for method in ("full", "random", "answer_attention"):
            with self.subTest(method=method):
                budget = 1 if method == "full" else 0.5
                tuned, meta = self.fit(method=method, budget=budget, steps=2)
                self.assertEqual(meta["n_retained_demo_tokens"], int(19*budget))
                self.assertEqual(len(meta["loss_per_step"]), 2)
                self.assertEqual(tuned.layers[0][0].shape[1], 2)
                self.assertTrue(all(p.grad is None for p in self.model.parameters()))
                self.assertTrue(ctkv.decode(self.model, TokenizerStub(), tuned, self.query, 3)[1])


if __name__ == "__main__":
    unittest.main()
