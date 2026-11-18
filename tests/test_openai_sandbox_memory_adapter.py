from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.openai_sandbox_memory_adapter import (
    PHASE_TWO_SELECTION_CHUNK_SIZE,
    _chat_completions_memory_phase_compatibility,
    _chunk_phase_two_selection,
    _inbox_agent_instructions,
    _INBOX_USER_PROMPT,
    _phase_two_evidence_aliases,
    _phase_two_workspace_snapshot,
    _reduce_phase_two_plans,
    _render_phase_two_plan,
)


def _selection() -> SimpleNamespace:
    return _selection_with_count(1)


def _selection_with_count(count: int) -> SimpleNamespace:
    selected = [
        SimpleNamespace(
            rollout_id=f"rollout-{index}",
            rollout_summary_file=f"rollout_summaries/rollout-{index}.md",
            updated_at=f"2026-08-{min(index + 9, 28):02d}T12:00:00+00:00",
            terminal_state="completed",
        )
        for index in range(1, count + 1)
    ]
    return SimpleNamespace(
        selected=selected,
        retained_rollout_ids={
            item.rollout_id for item in selected if int(item.rollout_id[8:]) % 2 == 0
        },
        removed=[
            SimpleNamespace(
                rollout_id="removed-rollout",
                rollout_summary_file="rollout_summaries/removed-rollout.md",
                updated_at="2026-07-01T12:00:00+00:00",
                terminal_state="completed",
            )
        ],
    )


class PhaseTwoPlanRendererTests(unittest.TestCase):
    def test_renders_only_claims_with_selected_evidence(self) -> None:
        plan = SimpleNamespace(
            entries=[
                SimpleNamespace(
                    kind="user_preference",
                    task_group="Shopping\n# injected heading",
                    claim="User prefers quiet products.\n## injected section",
                    keywords=["quiet", "shopping"],
                    evidence_id="rollout-1",
                ),
                SimpleNamespace(
                    kind="other",
                    task_group="Unsupported",
                    claim="This must not survive.",
                    keywords=["unsupported"],
                    evidence_id="made-up",
                ),
            ]
        )

        memory_md, summary_md = _render_phase_two_plan(plan, _selection())

        self.assertIn("rollout_id=rollout-1", memory_md)
        self.assertNotIn("rollout_id=made-up", memory_md)
        self.assertNotIn("This must not survive", memory_md)
        self.assertIn("User prefers quiet products. ## injected section", memory_md)
        self.assertNotIn("\n## injected section", memory_md)
        self.assertIn("User prefers quiet products.", summary_md)
        self.assertIn("#### 2026-08-10", summary_md)

    def test_emits_minimal_files_when_no_claim_has_valid_evidence(self) -> None:
        plan = SimpleNamespace(
            entries=[
                SimpleNamespace(
                    kind="user_preference",
                    task_group="Shopping",
                    claim="Unsupported preference.",
                    keywords=["shopping"],
                    evidence_id="unknown",
                )
            ]
        )

        memory_md, summary_md = _render_phase_two_plan(plan, _selection())

        self.assertIn("No meaningful reusable learning was promoted", memory_md)
        self.assertIn("No stable user preferences were promoted", summary_md)
        self.assertNotIn("Unsupported preference", memory_md + summary_md)

    def test_reduce_preserves_more_than_twelve_valid_distinct_entries(self) -> None:
        selection = _selection_with_count(20)
        mapped_plans = []
        for offset in range(0, 20, PHASE_TWO_SELECTION_CHUNK_SIZE):
            mapped_plans.append(
                SimpleNamespace(
                    entries=[
                        SimpleNamespace(
                            kind=(
                                "external_fact" if index == 20 else "user_preference"
                            ),
                            task_group="Shopping",
                            claim=f"Preference claim {index}.",
                            keywords=[f"product-{index}", f"category-{index}"],
                            evidence_id=f"r{index:03d}",
                        )
                        for index in range(
                            offset + 1,
                            min(offset + PHASE_TWO_SELECTION_CHUNK_SIZE, 20) + 1,
                        )
                    ]
                )
            )
        mapped_plans.append(
            SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        kind="user_preference",
                        task_group="Duplicate",
                        claim="Preference claim 1.",
                        keywords=["duplicate"],
                        evidence_id="r002",
                    ),
                    SimpleNamespace(
                        kind="user_preference",
                        task_group="Unsupported",
                        claim="Unsupported claim.",
                        keywords=["unsupported"],
                        evidence_id="r999",
                    ),
                ]
            )
        )

        reduced = _reduce_phase_two_plans(mapped_plans, selection)
        memory_md, summary_md = _render_phase_two_plan(reduced, selection)

        self.assertEqual(len(reduced.entries), 20)
        self.assertEqual(reduced.entries[-1].evidence_id, "rollout-20")
        self.assertEqual(reduced.entries[-1].kind, "external_fact")
        self.assertIn("Preference claim 20.", memory_md)
        self.assertIn("## Reusable knowledge", memory_md)
        self.assertIn("Preference claim 20.", summary_md)
        self.assertNotIn("Unsupported claim", memory_md + summary_md)


class PhaseTwoChunkingTests(unittest.TestCase):
    def test_chunks_twenty_rollouts_with_globally_stable_aliases(self) -> None:
        selection = _selection_with_count(20)

        chunks = _chunk_phase_two_selection(selection)
        aliases = _phase_two_evidence_aliases(selection)

        self.assertEqual([len(chunk.selected) for chunk in chunks], [8, 8, 4])
        self.assertEqual(aliases["r001"], "rollout-1")
        self.assertEqual(aliases["r009"], "rollout-9")
        self.assertEqual(aliases["r020"], "rollout-20")
        self.assertEqual(
            chunks[1].retained_rollout_ids,
            {"rollout-10", "rollout-12", "rollout-14", "rollout-16"},
        )
        self.assertEqual(chunks[0].removed, selection.removed)

    def test_snapshot_prioritizes_every_shard_evidence_file(self) -> None:
        selection = _selection_with_count(20)
        chunk = _chunk_phase_two_selection(selection)[1]
        aliases = _phase_two_evidence_aliases(selection)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            memories = root / "memories"
            (memories / "raw_memories").mkdir(parents=True)
            (memories / "rollout_summaries").mkdir(parents=True)
            for item in chunk.selected:
                (memories / "raw_memories" / f"{item.rollout_id}.md").write_text(
                    f"raw evidence for {item.rollout_id} " + "x" * 100,
                    encoding="utf-8",
                )
                (memories / item.rollout_summary_file).write_text(
                    f"summary evidence for {item.rollout_id} " + "y" * 100,
                    encoding="utf-8",
                )

            snapshot = json.loads(
                _phase_two_workspace_snapshot(
                    workspace_root=root,
                    memory_root="memories",
                    selection=chunk,
                    evidence_aliases=aliases,
                    char_budget=320,
                )
            )

        self.assertEqual(
            snapshot["evidence_aliases"],
            {f"r{index:03d}": f"rollout-{index}" for index in range(9, 17)},
        )
        paths = {item["path"] for item in snapshot["files"]}
        for index in range(9, 17):
            self.assertIn(f"memories/raw_memories/rollout-{index}.md", paths)
            self.assertIn(f"memories/rollout_summaries/rollout-{index}.md", paths)

    def test_snapshot_skips_missing_raw_file_without_losing_summary(self) -> None:
        selection = _selection()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            summary_path = root / "memories" / "rollout_summaries" / "rollout-1.md"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("summary only", encoding="utf-8")

            snapshot = json.loads(
                _phase_two_workspace_snapshot(
                    workspace_root=root,
                    memory_root="memories",
                    selection=selection,
                )
            )

        self.assertEqual(
            [item["path"] for item in snapshot["files"]],
            ["memories/rollout_summaries/rollout-1.md"],
        )

    def test_rejects_invalid_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _chunk_phase_two_selection(_selection(), chunk_size=0)


class PhaseTwoCompatibilityIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_two_maps_all_chunks_and_writes_all_entries(self) -> None:
        selection = _selection_with_count(20)
        calls: list[str] = []

        class FakeRunner:
            @staticmethod
            async def run(agent, prompt, *, run_config, max_turns):
                del agent, run_config, max_turns
                calls.append(prompt)
                snapshot = json.loads(
                    prompt.split(
                        "EMBEDDED WORKSPACE SNAPSHOT (JSON DATA):\n",
                        maxsplit=1,
                    )[1]
                )
                entries = [
                    SimpleNamespace(
                        kind="external_fact",
                        task_group="Inbox evidence",
                        claim=f"Claim retained from {rollout_id}.",
                        keywords=[rollout_id],
                        evidence_id=alias,
                    )
                    for alias, rollout_id in snapshot["evidence_aliases"].items()
                ]
                plan = SimpleNamespace(entries=entries)
                return SimpleNamespace(
                    final_output_as=lambda output_type, raise_if_incorrect_type: plan
                )

        async def original_phase_one(**kwargs):
            del kwargs

        async def original_phase_two(**kwargs):
            del kwargs

        memory_manager = SimpleNamespace(
            run_phase_one=original_phase_one,
            run_phase_two=original_phase_two,
        )
        sdk = SimpleNamespace(
            Agent=lambda **kwargs: kwargs,
            MemoryConsolidationPlan=object,
            MemoryManager=memory_manager,
            RolloutExtractionArtifacts=object,
            Runner=FakeRunner,
            render_memory_consolidation_prompt=lambda **kwargs: (
                f"official prompt for {len(kwargs['selection'].selected)} inputs"
            ),
            render_rollout_extraction_prompt=lambda **kwargs: "phase one",
        )
        config = SimpleNamespace(
            phase_two_model="model",
            phase_two_model_settings=None,
            extra_prompt=None,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            memories = root / "memories"
            (memories / "raw_memories").mkdir(parents=True)
            (memories / "rollout_summaries").mkdir(parents=True)
            for item in selection.selected:
                (memories / "raw_memories" / f"{item.rollout_id}.md").write_text(
                    f"raw evidence for {item.rollout_id}",
                    encoding="utf-8",
                )
                (memories / item.rollout_summary_file).write_text(
                    f"summary evidence for {item.rollout_id}",
                    encoding="utf-8",
                )

            with _chat_completions_memory_phase_compatibility(
                sdk,
                workspace_root=root,
            ):
                await sdk.MemoryManager.run_phase_two(
                    config=config,
                    memory_root="memories",
                    selection=selection,
                    run_config=object(),
                )

            memory_md = (memories / "MEMORY.md").read_text(encoding="utf-8")
            summary_md = (memories / "memory_summary.md").read_text(encoding="utf-8")

        self.assertEqual(len(calls), 3)
        self.assertIn("evidence shard 1 of 3", calls[0])
        self.assertIn("evidence shard 3 of 3", calls[2])
        self.assertIn("Assess every evidence alias independently", calls[0])
        self.assertIn("eligible as external_fact", calls[0])
        for index in range(1, 21):
            self.assertIn(f"Claim retained from rollout-{index}.", memory_md)
            self.assertIn(f"Claim retained from rollout-{index}.", summary_md)
        self.assertIs(sdk.MemoryManager.run_phase_two, original_phase_two)


class InboxPromptBoundaryTests(unittest.TestCase):
    def test_path_and_command_stay_out_of_user_message(self) -> None:
        root = Path("/tmp/openai-memory-test")
        batch_path = root / "inbox" / "batch.jsonl"

        instructions = _inbox_agent_instructions(
            batch_path=batch_path,
            workspace_root=root,
        )

        self.assertEqual(_INBOX_USER_PROMPT, "Process the assigned inbox batch.")
        self.assertNotIn("cat ", _INBOX_USER_PROMPT)
        self.assertNotIn(batch_path.as_posix(), _INBOX_USER_PROMPT)
        self.assertIn(f"cmd=cat {batch_path.as_posix()}", instructions)
        self.assertIn("exact named entities", instructions)
        self.assertNotIn("target", instructions.casefold())


if __name__ == "__main__":
    unittest.main()
