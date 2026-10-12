"""Aligned v1 acceptance frozen before the boundary-safe training run.

Replace only invalid CAV0 171/172 by the closest adjacent safe predictions
136/137 before the first shared boundary (188), with all 50 future frames.
Do not select samples using prediction scores. Legacy windows stay untouched.
"""
ALIGNED_FORECAST_WINDOW = (
    {"scene_id": "testoutput_CAV_data_2022-03-17-10-43-13", "receiver_id": 0,
     "start_frame": 136, "end_frame": 137},
    {"scene_id": "testoutput_CAV_data_2022-03-17-10-43-13", "receiver_id": 1,
     "start_frame": 75, "end_frame": 76},
)
ALIGNED_ACCEPTANCE_STEPS = 400
ALIGNED_ACCEPTANCE_GLOBAL_BATCH = 4
