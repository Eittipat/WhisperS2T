import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from whisper_s2t.backends.tensorrt import engine_builder as builder


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(builder.importlib.metadata, "version", return_value="1.2.1"),
            patch.object(
                builder.torch.cuda, "get_device_capability", return_value=(8, 9)
            ),
            patch.object(
                builder.torch.cuda, "get_device_name", return_value="RTX 4070 Ti"
            ),
        ]
        for mock in self.patches:
            mock.start()
            self.addCleanup(mock.stop)

    def test_decoder_position_limit_and_invalid_options(self):
        config = builder.TRTBuilderConfig(max_input_len=4, max_output_len=448)
        self.assertEqual(config.max_input_len + config.max_output_len, 448)
        for kwargs in (
            {"max_input_len": 448},
            {"max_batch_size": 0},
            {"dtype": "int8"},
            {"quantization": "fp8"},
            {"use_weight_only_enc": True},
        ):
            with self.assertRaises(ValueError):
                builder.TRTBuilderConfig(**kwargs)

    def test_cache_identity_includes_gpu_and_build_limits(self):
        original = builder.TRTBuilderConfig().identifier()
        self.assertNotEqual(
            original, builder.TRTBuilderConfig(max_batch_size=1).identifier()
        )
        self.assertNotEqual(
            original, builder.TRTBuilderConfig(quantization="int8").identifier()
        )
        with patch.object(builder.torch.cuda, "get_device_name", return_value="L4"):
            self.assertNotEqual(original, builder.TRTBuilderConfig().identifier())

    def test_loader_rejects_legacy_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trt_build_args.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Legacy"):
                builder.load_trt_build_config(tmp)

    def test_loader_warns_for_different_gpu_model_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trt_build_args.json"
            config = vars(builder.TRTBuilderConfig())
            config["gpu_name"] = "L4"
            original = json.dumps(config)
            path.write_text(original)
            with self.assertLogs(builder.__name__, level="WARNING") as logs:
                loaded = builder.load_trt_build_config(tmp)
            self.assertEqual(vars(loaded), config)
            self.assertEqual(path.read_text(), original)
            self.assertIn("L4", logs.output[0])
            self.assertIn("RTX 4070 Ti", logs.output[0])
            self.assertIn("continuing", logs.output[0])

    def test_loader_does_not_warn_for_matching_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = vars(builder.TRTBuilderConfig())
            (Path(tmp) / "trt_build_args.json").write_text(json.dumps(config))
            with self.assertNoLogs(builder.__name__, level="WARNING"):
                self.assertEqual(vars(builder.load_trt_build_config(tmp)), config)

    def test_gpu_name_warning_does_not_relax_architecture_or_version_checks(self):
        cases = (
            ("cuda_compute_capability", [8, 0], "GPU architecture"),
            ("tensorrt_llm_version", "0.8.0", "Legacy"),
            ("tensorrt_version", "other", "TensorRT version"),
            ("torch_version", "other", "PyTorch version"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trt_build_args.json"
            for field, value, message in cases:
                with self.subTest(field=field):
                    config = vars(builder.TRTBuilderConfig())
                    config.update(gpu_name="L4")
                    config[field] = value
                    path.write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, message):
                        builder.load_trt_build_config(tmp)
            config = vars(builder.TRTBuilderConfig())
            path.write_text(json.dumps(config))
            with patch.object(builder.importlib.metadata, "version", return_value="other"):
                with self.assertRaisesRegex(ValueError, "Runtime version"):
                    builder.load_trt_build_config(tmp)

    def test_int8_converter_receives_weight_only_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "small.pt"
            checkpoint.touch()
            with patch.object(builder, "SAVE_DIR", tmp), patch.object(
                builder, "download_model", return_value=(str(checkpoint), "unused")
            ), patch.object(
                builder.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "convert"),
            ) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    builder.build_trt_engine(
                        "small", builder.TRTBuilderConfig(quantization="int8")
                    )
                command = run.call_args.args[0]
                self.assertIn("--use_weight_only", command)
                self.assertEqual(
                    command[command.index("--weight_only_precision") + 1], "int8"
                )

    def test_failed_conversion_does_not_publish_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = builder.TRTBuilderConfig()
            checkpoint = Path(tmp) / "small.pt"
            checkpoint.touch()
            tokenizer = Path(tmp) / "tokenizer.json"
            tokenizer.write_text("{}")
            with patch.object(builder, "SAVE_DIR", tmp), patch.object(
                builder,
                "download_model",
                return_value=(str(checkpoint), str(tokenizer)),
            ), patch.object(
                builder.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "convert"),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    builder.build_trt_engine("small", config)
            self.assertFalse((Path(tmp) / "small" / config.identifier()).exists())
            self.assertEqual(list((Path(tmp) / "small").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
