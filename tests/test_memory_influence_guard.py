from __future__ import annotations

import unittest

from memory_influence_guard import (
    GuardDisposition,
    apply_strict_counterfactual_guard,
)


class StrictCounterfactualGuardTests(unittest.TestCase):
    def test_stable_memory_proposal_is_allowed(self) -> None:
        result = apply_strict_counterfactual_guard(
            baseline_choice_id="catalog-best",
            memory_choice_id="catalog-best",
            memory_was_read=True,
        )

        self.assertEqual(result.final_choice_id, "catalog-best")
        self.assertEqual(result.disposition, GuardDisposition.STABLE)
        self.assertFalse(result.memory_induced_change)
        self.assertTrue(result.action_authorized)

    def test_memory_induced_change_requires_confirmation(self) -> None:
        result = apply_strict_counterfactual_guard(
            baseline_choice_id="catalog-best",
            memory_choice_id="attacker-target",
            memory_was_read=True,
        )

        self.assertEqual(result.final_choice_id, "catalog-best")
        self.assertEqual(
            result.disposition,
            GuardDisposition.CONFIRMATION_REQUIRED,
        )
        self.assertTrue(result.memory_induced_change)
        self.assertTrue(result.confirmation_required)
        self.assertFalse(result.action_authorized)

    def test_noninteractive_mode_falls_back_to_no_memory_choice(self) -> None:
        result = apply_strict_counterfactual_guard(
            baseline_choice_id="catalog-best",
            memory_choice_id="attacker-target",
            memory_was_read=True,
            interaction_available=False,
        )

        self.assertEqual(result.final_choice_id, "catalog-best")
        self.assertEqual(
            result.disposition,
            GuardDisposition.FALLBACK_NO_INTERACTION,
        )
        self.assertTrue(result.action_authorized)

    def test_current_user_confirmation_recovers_legitimate_override(self) -> None:
        result = apply_strict_counterfactual_guard(
            baseline_choice_id="catalog-best",
            memory_choice_id="remembered-preference",
            memory_was_read=True,
            current_user_choice_id="remembered-preference",
        )

        self.assertEqual(result.final_choice_id, "remembered-preference")
        self.assertEqual(
            result.disposition,
            GuardDisposition.CURRENT_USER_CONFIRMED,
        )
        self.assertTrue(result.action_authorized)
        self.assertFalse(result.confirmation_required)

    def test_current_user_can_correct_both_memory_and_baseline(self) -> None:
        result = apply_strict_counterfactual_guard(
            baseline_choice_id="catalog-best",
            memory_choice_id="attacker-target",
            memory_was_read=True,
            current_user_choice_id="actual-preference",
        )

        self.assertEqual(result.final_choice_id, "actual-preference")
        self.assertTrue(result.memory_induced_change)

    def test_no_memory_path_cannot_claim_a_different_choice(self) -> None:
        with self.assertRaises(ValueError):
            apply_strict_counterfactual_guard(
                baseline_choice_id="one",
                memory_choice_id="two",
                memory_was_read=False,
            )

    def test_every_unconfirmed_memory_flip_uses_baseline(self) -> None:
        for index in range(100):
            baseline = f"baseline-{index}"
            attack = f"attack-{index}"
            result = apply_strict_counterfactual_guard(
                baseline_choice_id=baseline,
                memory_choice_id=attack,
                memory_was_read=True,
                interaction_available=bool(index % 2),
            )
            self.assertEqual(result.final_choice_id, baseline)
            self.assertNotEqual(result.final_choice_id, attack)


if __name__ == "__main__":
    unittest.main()
