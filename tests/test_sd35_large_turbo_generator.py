from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "data_generation" / "scripts"
MODULE_PATH = SCRIPTS_DIR / "sd35_large_turbo_generator.py"


class FakePipeline:
    last_from_pretrained_kwargs = None

    def __init__(self):
        self.cpu_offload_enabled = False
        self.group_offload_enabled = False
        self.group_offload_kwargs = None
        self.device = None
        self.attention_slicing_enabled = False
        self.vae_tiling_enabled = False
        self.vae_slicing_enabled = False

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_from_pretrained_kwargs = dict(kwargs)
        return cls()

    def enable_model_cpu_offload(self):
        self.cpu_offload_enabled = True

    def enable_group_offload(self, **kwargs):
        self.group_offload_enabled = True
        self.group_offload_kwargs = dict(kwargs)

    def to(self, device):
        self.device = device
        return self

    def enable_attention_slicing(self):
        self.attention_slicing_enabled = True

    def enable_vae_tiling(self):
        self.vae_tiling_enabled = True

    def enable_vae_slicing(self):
        self.vae_slicing_enabled = True


def load_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.device = lambda value: value

    class FakeGenerator:
        def __init__(self, device=None):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_torch.Generator = FakeGenerator
    fake_torch.dtype = object
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.StableDiffusion3Pipeline = FakePipeline
    fake_diffusers.SD3Transformer2DModel = type(
        "FakeTransformer",
        (),
        {"from_pretrained": classmethod(lambda cls, *args, **kwargs: object())},
    )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    fake_transformers.T5EncoderModel = type(
        "FakeT5EncoderModel",
        (),
        {"from_pretrained": classmethod(lambda cls, *args, **kwargs: object())},
    )

    old_modules = {}
    for name, module in {
        "torch": fake_torch,
        "diffusers": fake_diffusers,
        "transformers": fake_transformers,
    }.items():
        old_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = spec_from_file_location("test_sd35_large_turbo_generator_module", MODULE_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, old_modules


class SD35LargeTurboGeneratorTests(unittest.TestCase):
    def setUp(self):
        FakePipeline.last_from_pretrained_kwargs = None
        self.module, self.old_modules = load_module()

    def tearDown(self):
        if sys.path and sys.path[0] == str(SCRIPTS_DIR):
            sys.path.pop(0)
        for name, previous in self.old_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        sys.modules.pop("sd35_large_generator", None)
        sys.modules.pop("test_sd35_large_turbo_generator_module", None)

    def test_fit_24g_does_not_force_quantization_when_disabled(self):
        generator = self.module.SD35LargeTurboGenerator(
            model_path="/tmp/fake-sd35-turbo",
            runtime_profile="fit-24g",
            use_cpu_offload=False,
            prefer_quantized=False,
        )

        self.assertFalse(generator.prefer_quantized)

    def test_non_quantized_load_does_not_pass_null_transformer(self):
        self.module.SD35LargeTurboGenerator(
            model_path="/tmp/fake-sd35-turbo",
            runtime_profile="fit-24g",
            use_cpu_offload=False,
            prefer_quantized=False,
        )

        self.assertIsNotNone(FakePipeline.last_from_pretrained_kwargs)
        self.assertNotIn("transformer", FakePipeline.last_from_pretrained_kwargs)

    def test_turbo_does_not_enable_optional_memory_savers_by_default(self):
        generator = self.module.SD35LargeTurboGenerator(
            model_path="/tmp/fake-sd35-turbo",
            runtime_profile="fit-24g",
            use_cpu_offload=False,
            prefer_quantized=False,
        )

        self.assertFalse(generator.pipe.attention_slicing_enabled)
        self.assertFalse(generator.pipe.vae_tiling_enabled)
        self.assertFalse(generator.pipe.vae_slicing_enabled)

    def test_fit_24g_prefers_group_offload_when_available(self):
        generator = self.module.SD35LargeTurboGenerator(
            model_path="/tmp/fake-sd35-turbo",
            runtime_profile="fit-24g",
            use_cpu_offload=False,
            prefer_quantized=False,
        )

        self.assertTrue(generator.pipe.group_offload_enabled)
        self.assertFalse(generator.pipe.cpu_offload_enabled)
        self.assertEqual(generator.pipe.group_offload_kwargs["onload_device"], "cuda")
        self.assertEqual(generator.pipe.group_offload_kwargs["offload_type"], "block_level")
        self.assertEqual(generator.pipe.group_offload_kwargs["num_blocks_per_group"], 1)
        self.assertTrue(generator.pipe.group_offload_kwargs["use_stream"])

    def test_fit_24g_loads_8bit_text_encoder_3_when_available(self):
        self.module.SD35LargeTurboGenerator(
            model_path="/tmp/fake-sd35-turbo",
            runtime_profile="fit-24g",
            use_cpu_offload=False,
            prefer_quantized=False,
        )

        self.assertIn("text_encoder_3", FakePipeline.last_from_pretrained_kwargs)

    def test_fast_gpu_does_not_force_8bit_text_encoder_3(self):
        self.module.SD35LargeTurboGenerator(
            model_path="/tmp/fake-sd35-turbo",
            runtime_profile="fast-gpu",
            use_cpu_offload=False,
            prefer_quantized=False,
        )

        self.assertNotIn("text_encoder_3", FakePipeline.last_from_pretrained_kwargs)


if __name__ == "__main__":
    unittest.main()
