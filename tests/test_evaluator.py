"""Evaluator contract/resume tests use synthetic files and a tiny patched model.

No pretrained weights or network requests are made. Heavy checks are skipped
when the optional PyTorch stack is absent; config/data checks remain runnable.
"""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mms_eval import evaluator


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "numpy", "PIL"))


class EvaluatorContractTests(unittest.TestCase):
    def test_explicit_pretrained_defaults_and_no_default_alias(self):
        protocol, _ = evaluator._normalise_config({"classes": ["cat", "dog", "wild"]})
        self.assertEqual(protocol["weights"], "IMAGENET1K_V2")
        self.assertEqual(protocol["epochs"], 20)
        self.assertEqual(protocol["horizontal_flip_probability"], 0)
        protocol, _ = evaluator._normalise_config({"classes": ["a", "b"], "architecture": "vit_b_16"})
        self.assertEqual(protocol["weights"], "IMAGENET1K_V1")
        with self.assertRaises(ValueError):
            evaluator._normalise_config({"classes": ["a", "b"], "weights": "DEFAULT"})

    def test_invalid_classes_and_unknown_config_rejected(self):
        for config in ({"classes": ["a", "a"]}, {"classes": ["a"]}, {"classes": ["a", "b"], "epohcs": 5}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                evaluator._normalise_config(config)

    def test_operational_resume_changes_preserve_protocol(self):
        first, _ = evaluator._normalise_config({"classes": ["a", "b"], "device": "cpu", "epochs_per_call": 1})
        second, _ = evaluator._normalise_config({"classes": ["a", "b"], "device": "cuda:0", "num_workers": 2})
        self.assertEqual(first, second)
        changed, _ = evaluator._normalise_config({"classes": ["a", "b"], "head_lr": .5})
        self.assertNotEqual(evaluator._json_hash(first), evaluator._json_hash(changed))

    def test_config_policy_is_copied(self):
        config = {"classes": ["a", "b"], "image_policy": {"canonical_size": [256, 256]}}
        protocol, _ = evaluator._normalise_config(config)
        config["image_policy"]["canonical_size"][0] = 1
        self.assertEqual(protocol["image_policy"]["canonical_size"], [256, 256])

    def test_split_content_overlap_detected_even_under_new_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contents = [b"a", b"b", b"a", b"d"]
            rows = []
            for i, content in enumerate(contents):
                path = root / f"{i}.bin"
                path.write_bytes(content)
                rows.append({"path": str(path), "image_id": str(i), "label": i % 2})
            with self.assertRaisesRegex(ValueError, "overlap by sha256"):
                evaluator._prepare_data(rows[:2], rows[2:], ["a", "b"])

    def test_content_hash_changes_and_clone_path_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for i in range(2):
                path = root / f"{i}.bin"
                path.write_bytes(str(i).encode())
                rows.append({"path": str(path), "image_id": str(i), "label": i})
            _, original = evaluator._validate_records(rows, ["a", "b"], "Rfit")
            cloned = []
            for row in reversed(rows):
                target = root / f"copy_{row['image_id']}.bin"
                target.write_bytes(Path(row["path"]).read_bytes())
                cloned.append({**row, "path": str(target)})
            _, copied = evaluator._validate_records(cloned, ["a", "b"], "Rfit")
            self.assertEqual(original, copied)
            Path(cloned[0]["path"]).write_bytes(b"changed")
            _, modified = evaluator._validate_records(cloned, ["a", "b"], "Rfit")
            self.assertNotEqual(original, modified)


@unittest.skipUnless(HAS_TORCH, "PyTorch model extra is not installed")
class EvaluatorTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        from PIL import Image

        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.records = []
        for i in range(10):
            mode = "L" if i == 8 else "RGB"
            color = 10 + i * 17 if mode == "L" else (i * 17, 255 - i * 17, 20 + i)
            path = self.root / f"image_{i}.png"
            Image.new(mode, (16 + i, 30 + i), color=color).save(path)
            self.records.append({"path": str(path), "label": i % 2, "image_id": f"id_{i}"})
        self.train, self.val = self.records[:6], self.records[6:]
        self.config = {
            "classes": ["a", "b"], "weights": None, "epochs": 2,
            "micro_batch_size": 2, "effective_batch_size": 4,
            "seed": 17, "device": "cpu", "image_policy": {},
            "horizontal_flip_probability": 0.5,
        }

        def tiny_builder(architecture, weights, num_classes):
            self.assertIsNone(weights, "Test must never download weights")
            model = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
                torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, num_classes),
            )
            return model, model[-1]

        self.builder = mock.patch.object(evaluator, "_build_model", side_effect=tiny_builder)
        self.builder.start()

    def tearDown(self):
        self.builder.stop()
        self.torch.set_num_threads(self.previous_threads)
        self.directory.cleanup()

    def test_train_predict_freeze_and_resume_match_uninterrupted(self):
        full = evaluator.train_evaluator(self.train, self.val, self.root / "full", config=self.config)
        paused = evaluator.train_evaluator(
            self.train, self.val, self.root / "resume", config={**self.config, "epochs_per_call": 1}
        )
        self.assertEqual(paused["status"], "paused")
        resumed = evaluator.train_evaluator(self.train, self.val, self.root / "resume", config=self.config)
        self.assertEqual(resumed["status"], "complete")
        one = evaluator._load_checkpoint(full["last_checkpoint_path"])
        two = evaluator._load_checkpoint(resumed["last_checkpoint_path"])
        for key in one["model_state_dict"]:
            self.assertTrue(self.torch.equal(one["model_state_dict"][key], two["model_state_dict"][key]))
        self.assertEqual(full["best_val_nll"], resumed["best_val_nll"])
        predictions = evaluator.predict_paths([x["path"] for x in self.val], full["checkpoint_path"], batch_size=3)
        self.assertEqual(predictions["logits"].shape, (4, 2))
        self.assertFalse(predictions["metadata"]["temperature_applied"])
        self.assertEqual(predictions["metadata"]["classes"], ["a", "b"])
        self.assertEqual(len(predictions["metadata"]["checkpoint_sha256"]), 64)
        self.assertIsNone(predictions["metadata"]["preprocessing"]["crop"])
        again = evaluator.predict_paths([self.val[0]["path"]], full["checkpoint_path"])
        import numpy as np
        np.testing.assert_allclose(predictions["logits"][0], again["logits"][0], rtol=1e-6, atol=1e-6)
        empty = evaluator.predict_paths([], full["checkpoint_path"])
        self.assertEqual(empty["logits"].shape, (0, 2))
        metadata = json.loads(Path(full["metadata_path"]).read_text())
        self.assertEqual(metadata["best_epoch"], full["best_epoch"])

    def test_inference_disables_tf32_and_restores_callers_flags(self):
        run = evaluator.train_evaluator(self.train, self.val, self.root / "precision", config=self.config)
        torch = self.torch
        original_forward = torch.nn.Sequential.forward
        old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        def checked_forward(model, value):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.is_autocast_enabled())
            return original_forward(model, value)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with mock.patch.object(torch.nn.Sequential, "forward", new=checked_forward):
                evaluator.predict_paths([self.val[0]["path"]], run["checkpoint_path"])
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def test_prediction_session_reuses_model_across_feature_chunks(self):
        run = evaluator.train_evaluator(self.train, self.val, self.root / 'session', config=self.config)
        session = evaluator.PredictionSession()
        paths = [r['path'] for r in self.val]
        with mock.patch.object(evaluator, '_load_checkpoint', wraps=evaluator._load_checkpoint) as load:
            a = evaluator.predict_paths(paths[:2], run['checkpoint_path'], session=session)
            b = evaluator.predict_paths(paths[2:], run['checkpoint_path'], session=session)
            self.assertEqual(load.call_count, 1)
        import numpy as np
        full = evaluator.predict_paths(paths, run['checkpoint_path'], batch_size=2)
        np.testing.assert_array_equal(np.concatenate([a['logits'], b['logits']]), full['logits'])

    def test_changed_data_or_config_cannot_overwrite_existing_weights(self):
        run = evaluator.train_evaluator(self.train, self.val, self.root / "run", config=self.config)
        before = Path(run["checkpoint_path"]).read_bytes()
        with self.assertRaisesRegex(ValueError, "different config/data"):
            evaluator.train_evaluator(self.train, self.val, self.root / "run", config={**self.config, "seed": 9})
        changed = copy.deepcopy(self.train)
        changed[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "different config/data"):
            evaluator.train_evaluator(changed, self.val, self.root / "run", config=self.config)
        with self.assertRaises(FileExistsError):
            evaluator.train_evaluator(self.train, self.val, self.root / "run", config={**self.config, "resume": False})
        self.assertEqual(before, Path(run["checkpoint_path"]).read_bytes())

    def test_corrupt_checkpoint_metadata_is_rejected(self):
        run = evaluator.train_evaluator(self.train, self.val, self.root / "run", config=self.config)
        state = evaluator._load_checkpoint(run["checkpoint_path"])
        state["metadata"]["classes"] = ["b", "a"]
        path = self.root / "tampered.pt"
        self.torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            evaluator.predict_paths([], path)


if __name__ == "__main__":
    unittest.main()
