import unittest
from types import SimpleNamespace
from gain_router.local_preference import generation_stop_ids, generation_finish


class LocalPreferenceStopTests(unittest.TestCase):
    def test_chat_eos_and_base_eos_both_stop(self):
        ids = generation_stop_ids(SimpleNamespace(eos_token_id=248046), SimpleNamespace(eos_token_id=248044))
        self.assertEqual(ids, [248046, 248044])
        for last in ids:
            self.assertEqual(generation_finish([123, last], ids, 2), "stop")
        self.assertEqual(generation_finish([123, 456], ids, 2), "length")

    def test_list_eos_deduplicated_and_missing_rejected(self):
        self.assertEqual(generation_stop_ids(SimpleNamespace(eos_token_id=7), SimpleNamespace(eos_token_id=[7, 8])), [7, 8])
        with self.assertRaises(ValueError):
            generation_stop_ids(SimpleNamespace(eos_token_id=None), SimpleNamespace(eos_token_id=None))


if __name__ == "__main__":
    unittest.main()
