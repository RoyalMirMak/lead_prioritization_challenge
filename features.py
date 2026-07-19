import pandas as pd
from config import EVENT_FEATURE_COLS, EVENT_COUNT_COLS


def _safe_divide(numerator, denominator, fill_value=0.0):
    if denominator == 0:
        return fill_value
    return numerator / denominator


def _calculate_hours_since(reference_ts, event_ts):
    delta = reference_ts - event_ts
    if hasattr(delta, 'dt'):
        return delta.dt.total_seconds() / 3600
    return delta.total_seconds() / 3600


def _fill_missing_event_types(event_counts):
    for event_type in ['item_view', 'search', 'favorite', 'chat_open', 'call_click']:
        if event_type not in event_counts.columns:
            event_counts[event_type] = 0
    return event_counts


def _fill_missing_context_counts(event_counts, ctx_cols):
    for col in ctx_cols:
        col_name = f'events_ctx_{col}'
        if col_name not in event_counts.columns:
            event_counts[col_name] = 0
    return event_counts


def _extract_last_event_features(events):
    last_events = events.sort_values('event_ts').groupby('lead_id').last().reset_index()
    return last_events[['lead_id', 'event_type', 'ctx_seq']].rename(columns={
        'event_type': 'last_event_type',
        'ctx_seq': 'last_ctx_seq'
    })


def _extract_top_context_recency(events, top_contexts=['c03', 'c07', 'c05']):
    top_ctx_events = events[events['ctx_seq'].isin(top_contexts)].copy()
    
    if len(top_ctx_events) == 0:
        result = pd.DataFrame({'lead_id': events['lead_id'].unique()})
        for ctx in top_contexts:
            result[f'hours_since_last_{ctx}'] = 9999.0
        return result
    
    last_ctx_times = top_ctx_events.sort_values('event_ts').groupby('lead_id')[['ctx_seq', 'event_ts']].last()
    last_ctx_times = last_ctx_times.reset_index()
    last_ctx_times = last_ctx_times.pivot(index='lead_id', columns='ctx_seq', values='event_ts').reset_index()
    last_ctx_times.columns.name = None
    
    for ctx in top_contexts:
        if ctx not in last_ctx_times.columns:
            last_ctx_times[ctx] = pd.NaT
        last_ctx_times = last_ctx_times.rename(columns={ctx: f'last_{ctx}_ts'})
    
    return last_ctx_times


def _extract_24h_window_features(events, assignment_min):
    events_with_min = events.merge(assignment_min, on='lead_id', how='left')
    
    events_24h = events_with_min[
        (events_with_min['assignment_ts_min'] - events_with_min['event_ts']).dt.total_seconds() <= 86400
    ].copy()
    
    events_24h_counts = events_24h.groupby('lead_id').size().reset_index(name='events_total_24h')
    
    return events_24h_counts


def extract_event_features(events_df, assignment_df):
    events = events_df.copy()
    events['event_ts'] = pd.to_datetime(events['event_ts'])
    
    assignment_times = assignment_df[['lead_id', 'assignment_ts']].copy()
    assignment_times['assignment_ts'] = pd.to_datetime(assignment_times['assignment_ts'])
    
    events = events.merge(assignment_times, on='lead_id', how='inner')
    # отбрасываем все события, которые произошли позже или в момент назначения обращения
    events = events[events['event_ts'] < events['assignment_ts']]
    
    event_counts = events.groupby('lead_id')['event_type'].value_counts().unstack(fill_value=0).reset_index()
    event_counts = _fill_missing_event_types(event_counts)
    
    event_counts = event_counts.rename(columns={
        'item_view': 'events_item_view',
        'search': 'events_search',
        'favorite': 'events_favorite',
        'chat_open': 'events_chat_open',
        'call_click': 'events_call_click',
    })
    
    event_counts['events_total'] = sum(event_counts[col] for col in EVENT_COUNT_COLS[:-1])
    
    events['event_hour'] = events['event_ts'].dt.hour
    events['event_dow'] = events['event_ts'].dt.dayofweek
    events['is_weekend'] = (events['event_dow'] >= 5).astype(int)
    
    time_features = events.groupby('lead_id').agg(
        events_unique_types=('event_type', 'nunique'),
        events_active_days=('event_ts', lambda x: x.dt.date.nunique()),
        events_first_ts=('event_ts', 'min'),
        events_last_ts=('event_ts', 'max'),
        events_avg_price=('item_price_log', 'mean'),
        events_min_price=('item_price_log', 'min'),
        events_max_price=('item_price_log', 'max'),
        events_avg_slot=('src_slot', 'mean'),
        events_min_slot=('src_slot', 'min'),
        events_hour_mean=('event_hour', 'mean'),
        events_hour_std=('event_hour', 'std'),
        events_dow_mean=('event_dow', 'mean'),
        events_weekend_ratio=('is_weekend', 'mean'),
    ).reset_index()
    
    ctx_counts = events.groupby('lead_id')['ctx_seq'].value_counts().unstack(fill_value=0).reset_index()
    ctx_cols = [col for col in ctx_counts.columns if col != 'lead_id']
    ctx_counts = ctx_counts.rename(columns={col: f'events_ctx_{col}' for col in ctx_cols})
    
    last_events = _extract_last_event_features(events)
    top_ctx_times = _extract_top_context_recency(events)
    
    event_counts = event_counts.merge(time_features, on='lead_id', how='left')
    event_counts = event_counts.merge(ctx_counts, on='lead_id', how='left')
    event_counts = event_counts.merge(last_events, on='lead_id', how='left')
    event_counts = event_counts.merge(top_ctx_times, on='lead_id', how='left')
    
    event_counts['events_span_hours'] = _calculate_hours_since(
        event_counts['events_last_ts'], event_counts['events_first_ts']
    )
    
    assignment_min = assignment_times.groupby('lead_id')['assignment_ts'].min().reset_index()
    assignment_min.columns = ['lead_id', 'assignment_ts_min']
    event_counts = event_counts.merge(assignment_min, on='lead_id', how='left')
    
    event_counts['events_hours_since_last'] = _calculate_hours_since(
        event_counts['assignment_ts_min'], event_counts['events_last_ts']
    )
    
    event_counts['events_per_day'] = event_counts['events_total'] / (event_counts['events_active_days'] + 1)
    
    event_counts['events_favorite_to_view_ratio'] = (
        event_counts['events_favorite'] / (event_counts['events_item_view'] + 1)
    )
    event_counts['events_chat_to_call_ratio'] = (
        event_counts['events_chat_open'] / (event_counts['events_call_click'] + 1)
    )
    event_counts['events_search_to_view_ratio'] = (
        event_counts['events_search'] / (event_counts['events_item_view'] + 1)
    )
    
    event_counts['events_price_range'] = (
        event_counts['events_max_price'] - event_counts['events_min_price']
    )
    
    event_counts = _fill_missing_context_counts(event_counts, ctx_cols)
    
    for ctx in ['c03', 'c07', 'c05']:
        col_name = f'last_{ctx}_ts'
        if col_name in event_counts.columns:
            event_counts[f'hours_since_last_{ctx}'] = _calculate_hours_since(
                event_counts['assignment_ts_min'], event_counts[col_name]
            )
            event_counts[f'hours_since_last_{ctx}'] = event_counts[f'hours_since_last_{ctx}'].fillna(9999.0)
        else:
            event_counts[f'hours_since_last_{ctx}'] = 9999.0
    
    events_24h_counts = _extract_24h_window_features(events, assignment_min)
    event_counts = event_counts.merge(events_24h_counts, on='lead_id', how='left')
    event_counts['events_total_24h'] = event_counts['events_total_24h'].fillna(0)
    event_counts['recent_activity_ratio'] = (
        event_counts['events_total_24h'] / (event_counts['events_total'] + 1)
    )
    
    return_cols = ['lead_id'] + EVENT_FEATURE_COLS
    
    return event_counts[return_cols]
