import unittest

from rec_rocm.platforms import GIB, PLATFORMS, qualify


def stack(arch):
    p = PLATFORMS[arch]
    return p.torch, p.hip, p.triton


class QualifyTest(unittest.TestCase):
    def test_each_qualified_platform_is_accepted_with_its_own_limits(self):
        for arch, memory in (("gfx1201", 32 * GIB), ("gfx1151", 120 * GIB)):
            with self.subTest(arch):
                platform = qualify(arch, *stack(arch), 1, memory)
                self.assertIs(platform, PLATFORMS[arch])
        self.assertEqual(PLATFORMS["gfx1201"].max_context, 33536)
        self.assertGreaterEqual(PLATFORMS["gfx1151"].max_context, 32768 + 1024)

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "qualified only on gfx1201, gfx1151"):
            qualify("gfx1100", *stack("gfx1201"), 1, 64 * GIB)

    def test_stack_must_match_the_architecture(self):
        with self.assertRaisesRegex(RuntimeError, "pinned gfx1151"):
            qualify("gfx1151", *stack("gfx1201"), 1, 120 * GIB)
        for index in range(3):
            versions = list(stack("gfx1201"))
            versions[index] += ".x"
            with self.subTest(index), self.assertRaisesRegex(RuntimeError, "pinned gfx1201"):
                qualify("gfx1201", *versions, 1, 32 * GIB)

    def test_memory_floor_is_exclusive(self):
        for arch in PLATFORMS:
            floor = PLATFORMS[arch].min_memory
            with self.subTest(arch):
                with self.assertRaisesRegex(RuntimeError, "GiB of GPU-visible memory"):
                    qualify(arch, *stack(arch), 1, floor)
                qualify(arch, *stack(arch), 1, floor + 1)

    def test_exactly_one_device_is_required(self):
        for count in (0, 2):
            with self.subTest(count), self.assertRaisesRegex(RuntimeError, "Select one GPU"):
                qualify("gfx1201", *stack("gfx1201"), count, 32 * GIB)


if __name__ == "__main__":
    unittest.main()
