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

    def test_bucket_and_activation_checkpoint_flags(self) -> None:
        raw = {
            "distributed": {"world_size": 1, "backend": "gloo", "grad_sync_bucket_mb": 12},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {
                    "enabled": True,
                    "tp_size": 1,
                    "dp_size": 1,
                    "model_cls": "LLMModel",
                    "activation_checkpoint": True,
                },
                "decoder": {"enabled": False},
            },
            "pipeline": {"schedule": "gpipe", "num_micro_batches": 2},
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.distributed.grad_sync_bucket_mb, 12)
        self.assertTrue(cfg.stages["llm"].activation_checkpoint)
        self.assertTrue(cfg.training.io.enable_prefetch)

    def test_sequence_parallel_requires_tp(self) -> None:
        raw = {
            "distributed": {"world_size": 1, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {
                    "enabled": True,
                    "tp_size": 1,
                    "dp_size": 1,
                    "model_cls": "LLMModel",
                    "sequence_parallel": True,
                },
                "decoder": {"enabled": False},
            },
            "pipeline": {"schedule": "gpipe", "num_micro_batches": 1},
            "training": {
                "micro_batch_size": 2,
                "hidden_size": 64,
                "seq_len": 16,
                "num_attention_heads": 8,
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_zero_stage_parse(self) -> None:
        raw = {
            "distributed": {"world_size": 2, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 2, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {"schedule": "gpipe", "num_micro_batches": 2},
            "training": {
                "micro_batch_size": 2,
                "hidden_size": 64,
                "seq_len": 16,
                "optimizer": {
                    "type": "adamw",
                    "zero_stage": 1,
                },
            },
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.training.optimizer.zero_stage, 1)

    def test_zero_stage_validation(self) -> None:
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
                    "zero_stage": 2,
                },
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_loss_weights_parse(self) -> None:
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
                "loss_weights": {"text": 1.0, "image": 0.25, "audio": 0.5},
            },
        }
        cfg = RunConfig.from_dict(raw)
        self.assertAlmostEqual(cfg.training.loss_weights["image"], 0.25)
        self.assertAlmostEqual(cfg.training.loss_weights["audio"], 0.5)

    def test_loss_weights_invalid_key(self) -> None:
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
                "loss_weights": {"video": 1.0},
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_loss_weights_must_not_be_all_zero(self) -> None:
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
                "loss_weights": {"text": 0.0, "image": 0.0, "audio": 0.0},
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_deterministic_flag_parse(self) -> None:
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
                "deterministic": True,
            },
        }
        cfg = RunConfig.from_dict(raw)
        self.assertTrue(cfg.training.deterministic)

    def test_pipeline_transport_dtype_parse(self) -> None:
        raw = {
            "distributed": {"world_size": 1, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {
                "schedule": "gpipe",
                "num_micro_batches": 2,
                "transport_dtype": "bf16",
            },
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.pipeline.transport_dtype, "bf16")

    def test_pipeline_transport_dtype_validation(self) -> None:
        raw = {
            "distributed": {"world_size": 1, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {
                "schedule": "gpipe",
                "num_micro_batches": 2,
                "transport_dtype": "int8",
            },
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_zero1_bucket_mb_parse(self) -> None:
        raw = {
            "distributed": {"world_size": 2, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 2, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {"schedule": "gpipe", "num_micro_batches": 2},
            "training": {
                "micro_batch_size": 2,
                "hidden_size": 64,
                "seq_len": 16,
                "optimizer": {
                    "type": "adamw",
                    "zero_stage": 1,
                    "zero1_bucket_mb": 8.0,
                },
            },
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.training.optimizer.zero1_bucket_mb, 8.0)

    def test_zero1_bucket_mb_validation(self) -> None:
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
                    "zero_stage": 1,
                    "zero1_bucket_mb": -1,
                },
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_transport_tp_mode_parse(self) -> None:
        raw = {
            "distributed": {"world_size": 2, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "EncoderModel"},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {
                "schedule": "gpipe",
                "num_micro_batches": 2,
                "transport_tp_mode": "auto",
            },
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.pipeline.transport_tp_mode, "auto")

    def test_transport_tp_mode_direct_requires_equal_adjacent_tp(self) -> None:
        raw = {
            "distributed": {"world_size": 3, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "EncoderModel"},
                "llm": {"enabled": True, "tp_size": 2, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {
                "schedule": "gpipe",
                "num_micro_batches": 2,
                "transport_tp_mode": "direct",
            },
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_transport_tp_mode_validation(self) -> None:
        raw = {
            "distributed": {"world_size": 1, "backend": "gloo"},
            "stages": {
                "encoder": {"enabled": False},
                "llm": {"enabled": True, "tp_size": 1, "dp_size": 1, "model_cls": "LLMModel"},
                "decoder": {"enabled": False},
            },
            "pipeline": {
                "schedule": "gpipe",
                "num_micro_batches": 2,
                "transport_tp_mode": "invalid",
            },
            "training": {"micro_batch_size": 2, "hidden_size": 64, "seq_len": 16},
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_metrics_default_level_standard(self) -> None:
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
        self.assertEqual(cfg.training.metrics.level, "standard")
        self.assertFalse(cfg.training.metrics.groups.p2p_detail.enabled)
        self.assertFalse(cfg.training.metrics.groups.memory.enabled)
        self.assertTrue(cfg.training.metrics.groups.comm_summary.enabled)
        self.assertEqual(cfg.training.metrics.log_every_steps, 1)

    def test_metrics_group_override_parse(self) -> None:
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
                "metrics": {
                    "level": "minimal",
                    "rank_scope": "rank0",
                    "log_every_steps": 5,
                    "groups": {
                        "comm_summary": {"enabled": True, "every_n_steps": 3},
                        "p2p_detail": {"enabled": True, "every_n_steps": 7},
                    },
                },
            },
        }
        cfg = RunConfig.from_dict(raw)
        self.assertEqual(cfg.training.metrics.level, "minimal")
        self.assertEqual(cfg.training.metrics.rank_scope, "rank0")
        self.assertEqual(cfg.training.metrics.log_every_steps, 5)
        self.assertTrue(cfg.training.metrics.groups.comm_summary.enabled)
        self.assertEqual(cfg.training.metrics.groups.comm_summary.every_n_steps, 3)
        self.assertTrue(cfg.training.metrics.groups.p2p_detail.enabled)
        self.assertEqual(cfg.training.metrics.groups.p2p_detail.every_n_steps, 7)

    def test_metrics_validation(self) -> None:
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
                "metrics": {
                    "level": "invalid",
                    "groups": {"core": {"every_n_steps": 0}},
                },
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_metrics_every_n_validation(self) -> None:
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
                "metrics": {
                    "level": "standard",
                    "groups": {"core": {"every_n_steps": 0}},
                },
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)

    def test_metrics_rank_scope_validation(self) -> None:
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
                "metrics": {"rank_scope": "middle"},
            },
        }
        with self.assertRaises(ConfigError):
            RunConfig.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
