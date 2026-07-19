from pathlib import Path
import numpy as np
import optuna
import logging
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

RANDOM_STATE = 42

ROOT = Path(".")
DATA_DIR = ROOT / "data"

TARGET = "target"
ID_COLUMNS = {"lead_id", "user_id"}
TIME_COLUMNS = {"assignment_ts", "assignment_date"}
NON_FEATURE_COLUMNS = ID_COLUMNS | TIME_COLUMNS | {TARGET, "split"}

ENSEMBLE_SEEDS = [42, 179, 57, 91, 1543]

MODEL_CONFIG = {
    "iterations": 2000,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "early_stopping_rounds": 100,
    "task_type": "GPU",
    "devices": "0",
    "thread_count": 5,
    "depth": 7,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3.0,
    "random_seed": RANDOM_STATE,
}

EVENT_COUNT_COLS = [
    "events_item_view", "events_search", "events_favorite",
    "events_chat_open", "events_call_click", "events_total"
]

TEMPORAL_COLS = [
    "events_unique_types", "events_active_days", "events_span_hours",
    "events_hours_since_last", "events_per_day"
]

PRICE_COLS = [
    "events_avg_price", "events_min_price", "events_max_price",
    "events_price_range"
]

SLOT_COLS = ["events_avg_slot", "events_min_slot"]

RATIO_COLS = [
    "events_favorite_to_view_ratio", "events_chat_to_call_ratio",
    "events_search_to_view_ratio"
]

HOUR_DOW_COLS = [
    "events_hour_mean", "events_hour_std", "events_dow_mean",
    "events_weekend_ratio"
]

CTX_COUNT_COLS = [f"events_ctx_c{c:02d}" for c in range(1, 9)]

LAST_EVENT_COLS = ["last_event_type", "last_ctx_seq"]

RECENCY_COLS = ["hours_since_last_c03", "hours_since_last_c07", "hours_since_last_c05"]

WINDOW_24H_COLS = ["events_total_24h", "recent_activity_ratio"]

EVENT_FEATURE_COLS = (
    EVENT_COUNT_COLS + TEMPORAL_COLS + PRICE_COLS + SLOT_COLS + RATIO_COLS +
    HOUR_DOW_COLS + CTX_COUNT_COLS + LAST_EVENT_COLS + RECENCY_COLS + WINDOW_24H_COLS
)