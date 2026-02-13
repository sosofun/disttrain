from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

if torch is not None:
    from disttrain.config import load_config
    from disttrain.dist.groups import ProcessGroupManager
    from disttrain.dist.topology import Topology
    from disttrain.models.registry import build_stage_model
    from disttrain.pipeline.engine import TrainingEngine


@unittest.skipIf(torch is None, "torch is not installed in current environment")
class SmokeEngineTests(unittest.TestCase):
    def test_single_process_llm_only(self) -> None:
        cfg = load_config("configs/text_llm_only_local.yaml")
        topo = Topology(cfg, runtime_world_size=1, runtime_rank=0)
        model = build_stage_model(cfg.stages[topo.local_stage_name], cfg.training).to("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr)
        groups = ProcessGroupManager(topo)
        groups.create()

        engine = TrainingEngine(
            config=cfg,
            topology=topo,
            group_manager=groups,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )
        metrics = engine.run(max_steps=2)
        self.assertEqual(len(metrics), 2)
        self.assertGreater(metrics[0].step_time_sec, 0.0)
        self.assertGreater(metrics[0].loss, 0.0)


if __name__ == "__main__":
    unittest.main()
