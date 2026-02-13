from __future__ import annotations

import unittest

from disttrain.config import ConfigError, RunConfig
from disttrain.dist.topology import Topology


class ConfigTopologyTests(unittest.TestCase):
    def test_llm_only_world_size(self) -> None:
        raw = {
            "distributed": {"world_size": 1, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {"schedule": "gpipe", "num_micro_batches": 2},
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.expected_world_size, 1)
        topo = Topology(cfg, runtime_world_size=1, runtime_rank=0)
        self.assertEqual(topo.local_stage_name, "llm")
        self.assertEqual(topo.local_tp_index(), 0)
        self.assertEqual(topo.local_dp_index(), 0)

    def test_tri_stage_world_size(self) -> None:
        raw = {
            "distributed": {"world_size": 16, "backend": "gloo"},
            "stages": {
                "encoder": {
                    "enabled": True,
                    "tp_size": 2,
                    "dp_size": 2,
                    "model_cls": "EncoderModel",
                    "input_modalities": ["image"],
                },
                "llm": {"enabled": True, "tp_size": 4, "dp_size": 2, "model_cls": "LLMModel"},
                "decoder": {
                    "enabled": True,
                    "tp_size": 2,
                    "dp_size": 2,
                    "model_cls": "DecoderModel",
                    "output_modalities": ["audio"],
                },
            },
            "pipeline": {"schedule": "1f1b", "num_micro_batches": 8},
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.expected_world_size, 16)
        topo_rank0 = Topology(cfg, runtime_world_size=16, runtime_rank=0)
        topo_rank8 = Topology(cfg, runtime_world_size=16, runtime_rank=8)
        topo_rank15 = Topology(cfg, runtime_world_size=16, runtime_rank=15)
        self.assertEqual(topo_rank0.local_stage_name, "encoder")
        self.assertEqual(topo_rank8.local_stage_name, "llm")
        self.assertEqual(topo_rank15.local_stage_name, "decoder")

    def test_pipeline_micro_batches_must_cover_depth(self) -> None:
        raw = {
            "distributed": {"world_size": 3, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "EncoderModel"},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "DecoderModel"},
            },
            "pipeline": {"schedule": "1f1b", "num_micro_batches": 2},
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_stage_specific_lr_validation(self) -> None:
        raw = {
            "distributed": {"world_size": 1, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {"schedule": "gpipe", "num_micro_batches": 2},
            "training": {
                "micro_batch_size": 2,
                "hidden_size": 64,
                "seq_len": 16,
                "optimizer": {
                    "type": "adamw",
                    "stage_lrs": {"llm": 3e-4},
                },
            },
        }
        cfg = RunConfig.from_dict(raw)
        self.assertAlmostEqual(cfg.training.optimizer.stage_lrs["llm"], 3e-4)


if __name__ == "__main__":
    unittest.main()
