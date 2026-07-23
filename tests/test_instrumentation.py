"""Latency instrumentation (benchmark protocol §5).

Two properties matter most and are tested hardest:

1. **Attribution.** A method's auxiliary compute must not absorb the shared sampling
   cost, or the generator×family matrix reports the wrong shape entirely.
2. **Inertness.** With instrumentation off, nothing is recorded and no clock is read --
   otherwise the fork stops being a safe base for upstream merges.

Durations come from real ``sleep`` calls with generous tolerances: the point is to pin
which bucket time lands in, not to benchmark the timer.
"""

import time

import pytest

from TruthTorchLM.instrumentation.timing import (
    Stage,
    TimingRecord,
    begin_stage,
    capture,
    current_record,
    disable,
    enable,
    end_stage,
    is_enabled,
    record_metadata,
    stage,
)
from TruthTorchLM.instrumentation.stats import (
    overhead_matrix_row,
    sla_verdict,
    summarize,
)


class TestInertWhenDisabled:
    def test_no_record_outside_capture(self):
        disable()
        with stage(Stage.AUXILIARY_COMPUTE, "nothing"):
            pass
        assert current_record() is None
        assert is_enabled() is False

    def test_begin_stage_returns_none_when_disabled(self):
        disable()
        assert begin_stage(Stage.EXTRA_GENERATION, "x") is None
        end_stage(None)  # must be a safe no-op

    def test_capture_restores_previous_enabled_state(self):
        disable()
        with capture():
            assert is_enabled() is True
        assert is_enabled() is False

    def test_nested_capture_does_not_leak_the_inner_record(self):
        with capture() as outer:
            with capture() as inner:
                with stage(Stage.AUXILIARY_COMPUTE, "inner-only"):
                    pass
            with stage(Stage.AUXILIARY_COMPUTE, "outer-only"):
                pass
        assert [e.label for e in inner.events] == ["inner-only"]
        assert [e.label for e in outer.events] == ["outer-only"]


class TestStageAttribution:
    def test_stages_are_bucketed_separately(self):
        with capture() as rec:
            with stage(Stage.TARGET_GENERATION, "g"):
                time.sleep(0.05)
            with stage(Stage.EXTRA_GENERATION, "samples", n=5):
                time.sleep(0.03)
            with stage(Stage.AUXILIARY_COMPUTE, "nli"):
                time.sleep(0.02)

        assert rec.stage_ms(Stage.TARGET_GENERATION) == pytest.approx(50, abs=30)
        assert rec.stage_ms(Stage.EXTRA_GENERATION) == pytest.approx(30, abs=30)
        assert rec.stage_ms(Stage.AUXILIARY_COMPUTE) == pytest.approx(20, abs=30)

    def test_marginal_excludes_the_users_own_answer(self):
        """§5's core accounting rule: `g` is measured but never billed to the guardrail."""
        with capture() as rec:
            with stage(Stage.TARGET_GENERATION, "g"):
                time.sleep(0.08)
            with stage(Stage.EXTRA_GENERATION, "samples"):
                time.sleep(0.02)
            with stage(Stage.AUXILIARY_COMPUTE, "cluster"):
                time.sleep(0.02)

        assert rec.marginal_ms() < rec.total_ms()
        assert rec.marginal_ms() == pytest.approx(
            rec.stage_ms(Stage.EXTRA_GENERATION) + rec.stage_ms(Stage.AUXILIARY_COMPUTE)
        )
        assert rec.generation_baseline_ms() == pytest.approx(80, abs=40)

    def test_overhead_ratio_is_marginal_over_baseline(self):
        with capture() as rec:
            with stage(Stage.TARGET_GENERATION, "g"):
                time.sleep(0.10)
            with stage(Stage.EXTRA_GENERATION, "samples"):
                time.sleep(0.10)
        assert rec.overhead_ratio() == pytest.approx(1.0, rel=0.4)

    def test_overhead_ratio_is_none_without_a_measured_baseline(self):
        """The cached-harness case: Stage A produced the answer in an earlier pass."""
        with capture() as rec:
            with stage(Stage.AUXILIARY_COMPUTE, "cluster"):
                time.sleep(0.01)
        assert rec.overhead_ratio() is None

    def test_per_method_spans_are_separable_under_shared_sampling(self):
        """Protocol §6 shares one sample draw across all methods (upstream
        ``run_truth_methods``). Each method's own cost must still be recoverable, or a
        cheap scorer inherits an expensive one's sampling bill."""
        with capture() as rec:
            with stage(Stage.EXTRA_GENERATION, "shared_samples", n=10):
                time.sleep(0.20)
            with stage(Stage.AUXILIARY_COMPUTE, "VerbalizedConfidence"):
                pass  # a genuinely cheap scorer does almost no auxiliary work
            with stage(Stage.AUXILIARY_COMPUTE, "DiscreteSemanticEntropy"):
                time.sleep(0.15)

        cheap = rec.label_ms("VerbalizedConfidence")
        expensive = rec.label_ms("DiscreteSemanticEntropy")
        assert cheap < expensive
        assert cheap < 50  # nowhere near the 200ms of shared sampling
        assert expensive == pytest.approx(150, abs=80)

    def test_span_is_recorded_even_when_the_body_raises(self):
        """A method that dies after 40 seconds is a latency fact, not a lost measurement."""
        with capture() as rec:
            with pytest.raises(RuntimeError):
                with stage(Stage.AUXILIARY_COMPUTE, "explodes"):
                    time.sleep(0.01)
                    raise RuntimeError("boom")
        assert len(rec.events) == 1
        assert rec.events[0].duration_ms > 0

    def test_begin_end_pair_matches_the_context_manager(self):
        with capture() as rec:
            span = begin_stage(Stage.EXTRA_GENERATION, "loop", n=3, execution="serial")
            time.sleep(0.02)
            end_stage(span)
        assert len(rec.events) == 1
        assert rec.events[0].metadata == {"n": 3, "execution": "serial"}
        assert rec.events[0].duration_ms == pytest.approx(20, abs=30)

    def test_metadata_and_serialization(self):
        with capture(method="DSE", n=5) as rec:
            record_metadata(generator="gpt-4o-mini")
            with stage(Stage.AUXILIARY_COMPUTE, "nli", device="cuda"):
                pass
        payload = rec.as_dict()
        assert payload["metadata"] == {"method": "DSE", "n": 5, "generator": "gpt-4o-mini"}
        assert payload["events"][0]["metadata"] == {"device": "cuda"}
        assert set(payload) >= {"marginal_ms", "target_generation_ms", "overhead_ratio"}


class TestSummaryStats:
    def test_warmup_is_discarded_not_averaged_in(self):
        """The first calls pay for model loading and connection setup; folding them into
        p99 reports a number no steady-state user experiences."""
        samples = [5000.0] * 5 + [10.0] * 200
        summary = summarize(samples, warmup=5)
        assert summary["n_trials"] == 200
        assert summary["n_warmup_discarded"] == 5
        assert summary["p50_ms"] == pytest.approx(10.0)
        assert summary["max_ms"] == pytest.approx(10.0)

    def test_percentiles_track_the_tail(self):
        samples = [10.0] * 95 + [900.0] * 5
        summary = summarize(samples, warmup=0)
        assert summary["p50_ms"] == pytest.approx(10.0)
        assert summary["p99_ms"] > 500
        # A mean would have hidden this entirely -- which is why §5 forbids reporting one.
        assert summary["mean_ms"] < 100

    def test_underpowered_runs_are_flagged_not_silently_trusted(self):
        assert summarize([1.0] * 20, warmup=0)["underpowered"] is True
        assert summarize([1.0] * 150, warmup=0)["underpowered"] is False

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No latency samples"):
            summarize([], warmup=0)

    def test_samples_are_retained_for_downstream_cis(self):
        summary = summarize([1.0, 2.0, 3.0], warmup=0)
        assert summary["samples_ms"] == [1.0, 2.0, 3.0]


class TestSLAVerdict:
    def test_fast_method_passes_every_budget(self):
        verdict = sla_verdict(summarize([40.0] * 120, warmup=0))
        assert verdict["verdicts"] == {"500ms": True, "1000ms": True, "2000ms": True}
        assert verdict["tightest_budget_met_ms"] == 500.0

    def test_slow_method_fails_all_budgets(self):
        verdict = sla_verdict(summarize([5000.0] * 120, warmup=0))
        assert verdict["fits_any"] is False
        assert verdict["tightest_budget_met_ms"] is None

    def test_verdict_is_taken_on_p95_not_the_median(self):
        """Fast most of the time, stalling one request in twenty, is not 'fits 500ms'."""
        samples = [100.0] * 90 + [1500.0] * 10
        verdict = sla_verdict(summarize(samples, warmup=0))
        assert verdict["verdicts"]["500ms"] is False


class TestOverheadMatrix:
    def test_proxy_style_overhead_is_decoupled_from_the_generator(self):
        """§5's central finding, in numbers: a constant-ms proxy has a *smaller ratio* on
        a slower generator, while its absolute cost is unchanged."""
        proxy_cost_ms = 30.0
        rows = []
        for generator, g in (("fast-open-model", 200.0), ("slow-closed-lrm", 4000.0)):
            baseline = summarize([g] * 120, warmup=0)
            method = summarize([g + proxy_cost_ms] * 120, warmup=0)
            rows.append(overhead_matrix_row(generator, "P", "DisAAD", baseline, method))

        assert rows[0]["marginal_p50_ms"] == pytest.approx(rows[1]["marginal_p50_ms"])
        assert rows[0]["overhead_ratio_p50"] > rows[1]["overhead_ratio_p50"]
        assert all(r["sla"]["verdicts"]["500ms"] for r in rows)

    def test_consistency_style_overhead_is_coupled_to_the_generator(self):
        """The mirror image: a constant *ratio* whose absolute ms explodes on a slow
        target -- so SC x closed LRM is the worst cell in the matrix."""
        n = 5
        rows = []
        for generator, g in (("fast-open-model", 200.0), ("slow-closed-lrm", 4000.0)):
            baseline = summarize([g] * 120, warmup=0)
            method = summarize([g * n] * 120, warmup=0)
            rows.append(
                overhead_matrix_row(generator, "SC", "DSE", baseline, method, n=n)
            )

        assert rows[0]["overhead_ratio_p50"] == pytest.approx(rows[1]["overhead_ratio_p50"])
        assert rows[1]["marginal_p50_ms"] > 10 * rows[0]["marginal_p50_ms"]
        assert rows[0]["sla"]["verdicts"]["1000ms"] is True
        assert rows[1]["sla"]["fits_any"] is False

    def test_concurrency_can_rescue_a_multi_sample_method(self):
        """The other half of the §5 question: serial N=5 fails the budget, concurrent
        N=5 -- costing the tail of the parallel batch rather than the sum -- passes."""
        g = 200.0
        baseline = summarize([g] * 120, warmup=0)
        serial = overhead_matrix_row(
            "api", "SC", "DSE", baseline, summarize([g * 5] * 120, warmup=0),
            execution="serial", n=5,
        )
        concurrent = overhead_matrix_row(
            "api", "SC", "DSE", baseline, summarize([g * 1.3] * 120, warmup=0),
            execution="concurrent", n=5,
        )
        assert serial["sla"]["verdicts"]["500ms"] is False
        assert concurrent["sla"]["verdicts"]["500ms"] is True


def test_timing_record_is_constructible_standalone():
    rec = TimingRecord()
    assert rec.total_ms() == 0.0
    assert rec.marginal_ms() == 0.0
    assert rec.overhead_ratio() is None
