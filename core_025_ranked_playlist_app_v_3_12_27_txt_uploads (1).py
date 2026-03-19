# Updated app build prepared

The app has been updated so uploader fields that previously accepted only `.csv` now accept both `.csv` and `.txt` for these inputs:
- weights upload
- tie-pack upload
- rulepack upload
- 1-miss downranks upload
- rescue rules upload
- member score adjustments overlay upload

Updated downloadable file:
`/mnt/data/core025_ranked_playlist_app_v3_12_27_txt_uploads__2026-03-19.py`

Version string updated to `v3.12.27`.

Key code changes applied:
```python
weights_file_up = st.file_uploader("Upload weights CSV or TXT (optional)", type=["csv","txt"], key="weights_file_upload")
tiepack_file_up = st.file_uploader("Upload tie-pack CSV or TXT (optional)", type=["csv","txt"], key="tie_upload")
rp_up = st.file_uploader("Rulepack v3.1 (CSV or TXT) — member eliminators + optional overrides", type=["csv","txt"], key="mined_rulepack")
dr_up = st.file_uploader("1-miss downranks addon (CSV or TXT)", type=["csv","txt"], key="mined_downranks")
rs_up = st.file_uploader("Rescue rules (CSV or TXT) — per-event mined", type=["csv","txt"], key="mined_rescues")
overlay_up = st.file_uploader("Member score adjustments overlay CSV or TXT (additive)", type=["csv","txt"], key="member_score_overlay")
```