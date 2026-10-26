from types import SimpleNamespace

import pytest

from new_interest.experiments.history_length_ablation import stop_at_500, subset_metrics


def test_stop_keeps_full_scheduler_horizon_and_saves_exact_boundary():
    class Base:
        def add_callback(self, callback):
            self.callback = callback

    trainer = stop_at_500(Base)()
    callback = trainer.callback
    state = SimpleNamespace(global_step=0, max_steps=2025)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(None, state, control)
    state.global_step = 499
    callback.on_step_end(None, state, control)
    assert not control.should_training_stop
    state.global_step = 500
    callback.on_step_end(None, state, control)
    assert control.should_training_stop and control.should_save
    assert state.max_steps == 2025
    state.global_step = 501
    with pytest.raises(ValueError):
        callback.on_step_end(None, state, control)
    state.max_steps = 500
    with pytest.raises(ValueError):
        callback.on_train_begin(None, state, control)


def test_cohort_is_fixed_and_macro_is_class_equal_not_row_equal():
    records = [{"impression_id": str(i), "acceptable_subcategories": labels} for i, labels in
               enumerate([["A"], ["A"], ["B"], ["C"]])]
    predictions = [{"impression_id": str(i), "ranked_labels": labels} for i, labels in
                   enumerate([["A"], ["B"], ["B"], ["C"]])]
    result = subset_metrics(records, predictions, {"0", "1", "2"})
    assert result["records"] == 3
    assert result["supported_classes"] == 2
    assert result["metrics"]["interest_macro_recall_at_1"] == 0.75
    with pytest.raises(ValueError):
        subset_metrics(records, predictions, {"missing"})
