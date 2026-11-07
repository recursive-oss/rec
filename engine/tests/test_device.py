import unittest

from rec_mlx.kernels.device import profile_for_device


class DeviceProfileTests(unittest.TestCase):
    def test_m6_uses_generic_expert_path(self):
        profile = profile_for_device({"architecture": "applegpu_g18g", "device_name": "Apple M6"})
        self.assertEqual(profile.chip, "M6")
        self.assertEqual(profile.architecture, "applegpu_g18g")
        self.assertFalse(profile.specialize_mixed_experts)

    def test_m6_requires_matching_architecture(self):
        with self.assertRaises(ValueError):
            profile_for_device({"architecture": "applegpu_g17s", "device_name": "Apple M6"})

    def test_existing_profiles_are_preserved(self):
        for chip, architecture, workers, specialized in (
            ("M4 Pro", "applegpu_g16s", 36, False),
            ("M4 Max", "applegpu_g16s", 36, False),
            ("M5 Pro", "applegpu_g17s", 36, True),
            ("M5 Max", "applegpu_g17s", 76, True),
        ):
            with self.subTest(chip=chip):
                profile = profile_for_device(
                    {"architecture": architecture, "device_name": "Apple " + chip}
                )
                self.assertEqual(profile.workers, workers)
                self.assertEqual(profile.specialize_mixed_experts, specialized)
