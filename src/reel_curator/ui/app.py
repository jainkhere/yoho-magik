from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from reel_curator.analyzer import analyze_library
from reel_curator.cache import AnalysisCache
from reel_curator.config import AppConfig, load_config
from reel_curator.duplicates import duplicate_summary
from reel_curator.exports import export_contact_sheet, export_csv, export_json, export_videos
from reel_curator.face_priority import FacePriorityMatcher, normalized_reference_image
from reel_curator.scoring import SCORE_PRESETS, calculate_score

st.set_page_config(page_title="Reel Curator", layout="wide")
st.markdown(
    """
    <style>
    video {
        max-height: 260px;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    config = _sidebar_config()
    cache = AnalysisCache(config.cache_dir)

    st.title("Reel Curator")
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("Scan / Resume", type="primary", width="stretch"):
            with st.spinner("Analyzing videos locally..."):
                st.session_state["analyses"] = analyze_library(config, force=False)
    with col_b:
        if st.button("Force Rescan", width="stretch"):
            with st.spinner("Rebuilding analysis cache..."):
                st.session_state["analyses"] = analyze_library(config, force=True)

    analyses = st.session_state.get("analyses") or cache.all()
    if not analyses:
        st.info("Choose a folder and start a scan. Cached results will appear here after analysis.")
        return

    if not config.scan_landscape_videos:
        analyses = [item for item in analyses if item.metadata.orientation != "landscape"]
    _apply_score_preferences(analyses, config.scoring_weights)

    filtered = _filters(analyses)
    _summary(filtered, analyses)
    _exports(filtered, config)
    _gallery(filtered, cache)
    _duplicate_compare(filtered)
    _score_table(filtered)


def _sidebar_config() -> AppConfig:
    config = load_config()
    with st.sidebar:
        st.header("Library")
        input_folder = st.text_input("Video folder", value=str(config.input_folder))
        cache_dir = st.text_input("Cache folder", value=str(config.cache_dir))
        export_dir = st.text_input("Export folder", value=str(config.export_dir))
        workers = st.slider("Workers", 1, 12, config.workers)
        sample_frames = st.slider("Frames sampled per video", 3, 40, config.sample_frames)
        threshold = st.slider("Duplicate sensitivity", 0, 24, config.duplicate_hamming_threshold)
        enable_vision = st.toggle("Apple Vision", value=config.enable_apple_vision)
        scan_landscape = st.toggle(
            "Scan landscape videos",
            value=config.scan_landscape_videos,
            help=(
                "Turn this off to skip landscape clips during scan and hide cached "
                "landscape clips."
            ),
        )
        prioritize_people = st.toggle("Prioritize people", value=config.prioritize_people)
        face_threshold = st.slider(
            "Reference face match strictness",
            0.50,
            0.95,
            config.face_match_threshold,
            0.01,
        )
        _people_reference_uploader(Path(cache_dir).expanduser(), face_threshold)
        scoring_preset, scoring_weights = _scoring_controls(config)
    return AppConfig(
        input_folder=Path(input_folder).expanduser(),
        cache_dir=Path(cache_dir).expanduser(),
        export_dir=Path(export_dir).expanduser(),
        workers=workers,
        sample_frames=sample_frames,
        duplicate_hamming_threshold=threshold,
        enable_apple_vision=enable_vision,
        scan_landscape_videos=scan_landscape,
        prioritize_people=prioritize_people,
        face_match_threshold=face_threshold,
        scoring_preset=scoring_preset,
        scoring_weights=scoring_weights,
    )


def _people_reference_uploader(cache_dir: Path, face_threshold: float) -> None:
    st.divider()
    st.subheader("Priority People")
    matcher = FacePriorityMatcher(cache_dir / "priority_people", threshold=face_threshold)
    roster = matcher.roster()
    if roster:
        st.caption(f"{len(roster)} people prioritized")
        for name, paths in sorted(roster.items()):
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(f"{len(paths)} accepted reference photo(s)")
                columns = st.columns(3)
                for index, path in enumerate(paths[:3]):
                    image_bytes = normalized_reference_image(path)
                    columns[index % 3].image(image_bytes or str(path), width=58)
    else:
        st.caption("Add clear face photos to boost matching videos.")

    person_name = st.text_input("Person name", value="", placeholder="e.g. Minsi")
    uploads = st.file_uploader(
        "Reference face photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if st.button("Add reference faces", disabled=not uploads):
        accepted = 0
        rejected: list[str] = []
        for upload in uploads or []:
            result = matcher.add_reference_image(
                person_name or Path(upload.name).stem,
                upload.getvalue(),
                Path(upload.name).suffix,
            )
            if result.accepted:
                accepted += 1
            else:
                rejected.append(f"{upload.name}: {result.message}")
        if accepted:
            st.success(f"Added {accepted} reference photo(s). Force Rescan updates matches.")
        for message in rejected:
            st.warning(message)


def _scoring_controls(config: AppConfig) -> tuple[str, dict[str, float]]:
    st.divider()
    st.subheader("Scoring")
    preset_names = list(SCORE_PRESETS)
    default_index = (
        preset_names.index(config.scoring_preset)
        if config.scoring_preset in preset_names
        else 0
    )
    preset = st.selectbox("Preset", preset_names, index=default_index)
    preset_weights = SCORE_PRESETS[preset]

    with st.expander("Fine tune scoring"):
        st.caption("Weights are normalized automatically.")
        people = st.slider("People", 0, 50, int(preset_weights["people"]))
        image_quality = st.slider("Image quality", 0, 50, int(preset_weights["image_quality"]))
        stability = st.slider("Stability", 0, 50, int(preset_weights["stability"]))
        lighting = st.slider("Lighting", 0, 50, int(preset_weights["lighting"]))
        scenic = st.slider("Scenery", 0, 50, int(preset_weights["scenic_content"]))
        uniqueness = st.slider("Uniqueness", 0, 50, int(preset_weights["uniqueness"]))
        composition = st.slider("Composition", 0, 50, int(preset_weights["composition"]))
        reel = st.slider("Reel usefulness", 0, 50, int(preset_weights["reel_usefulness"]))

    weights = {
        "people": float(people),
        "image_quality": float(image_quality),
        "stability": float(stability),
        "lighting": float(lighting),
        "scenic_content": float(scenic),
        "uniqueness": float(uniqueness),
        "composition": float(composition),
        "reel_usefulness": float(reel),
    }
    return preset, weights


def _apply_score_preferences(analyses, weights: dict[str, float]) -> None:
    for item in analyses:
        item.score = calculate_score(
            metadata=item.metadata,
            brightness=item.brightness,
            sharpness=item.sharpness,
            stability=item.stability,
            motion=item.motion,
            tags=item.tags,
            uniqueness=item.score.uniqueness / 100.0,
            people_priority=item.priority_people_score,
            weights=weights,
        )


def _filters(analyses):
    st.subheader("Review")
    all_tags = sorted({tag for item in analyses for tag in item.tags})
    all_groups = sorted({item.duplicate_group for item in analyses if item.duplicate_group})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_score = st.slider("Minimum quality", 0, 100, 60)
        duration = st.slider("Duration seconds", 0, 180, (0, 60))
    with col2:
        tags = st.multiselect("Tags", all_tags)
        orientation = st.multiselect("Orientation", ["portrait", "landscape", "square"])
    with col3:
        resolution = st.text_input("Resolution contains", "")
        duplicate_mode = st.selectbox(
            "Duplicate groups",
            ["All", "Only duplicates", "Hide duplicates", "Representatives"],
        )
    with col4:
        groups = st.multiselect("Group IDs", all_groups)
        sort_by = st.selectbox(
            "Sort by",
            ["Quality", "Date", "Duration", "Scenic score", "Stability", "Brightness"],
        )

    filtered = []
    for item in analyses:
        if item.score.overall < min_score:
            continue
        if not duration[0] <= item.metadata.duration_seconds <= duration[1]:
            continue
        if tags and not set(tags).issubset(item.tags):
            continue
        if orientation and item.metadata.orientation not in orientation:
            continue
        if resolution and resolution not in item.metadata.resolution_label:
            continue
        if groups and item.duplicate_group not in groups:
            continue
        if duplicate_mode == "Only duplicates" and not item.duplicate_group:
            continue
        if (
            duplicate_mode == "Hide duplicates"
            and item.duplicate_group
            and not item.duplicate_representative
        ):
            continue
        if (
            duplicate_mode == "Representatives"
            and item.duplicate_group
            and not item.duplicate_representative
        ):
            continue
        filtered.append(item)

    key_map = {
        "Quality": lambda row: row.score.overall,
        "Date": lambda row: row.metadata.created_at or row.metadata.modified_time,
        "Duration": lambda row: row.metadata.duration_seconds,
        "Scenic score": lambda row: row.score.scenic_content,
        "Stability": lambda row: row.score.stability,
        "Brightness": lambda row: row.brightness,
    }
    return sorted(filtered, key=key_map[sort_by], reverse=True)


def _summary(filtered, analyses) -> None:
    groups = duplicate_summary(analyses)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Showing", len(filtered))
    c2.metric("Analyzed", len(analyses))
    c3.metric("Duplicate groups", len(groups))
    people_count = sum(1 for item in filtered if "people" in item.tags)
    c4.metric("People clips", people_count)


def _exports(analyses, config: AppConfig) -> None:
    with st.expander("Export"):
        col1, col2, col3, col4 = st.columns(4)
        top_n = col1.number_input("Top N", min_value=1, max_value=500, value=25)
        if col1.button("Export Top N"):
            path = export_videos(analyses, config.export_dir, "top_n", int(top_n))
            st.success(f"Copied videos to {path}")
        if col2.button("Export Favorites"):
            path = export_videos(analyses, config.export_dir, "favorites", int(top_n))
            st.success(f"Copied videos to {path}")
        if col2.button("Export Duplicate Reps"):
            path = export_videos(
                analyses, config.export_dir, "duplicate_representatives", int(top_n)
            )
            st.success(f"Copied videos to {path}")
        if col3.button("CSV / JSON"):
            csv_path = export_csv(analyses, config.export_dir / "metadata.csv")
            json_path = export_json(analyses, config.export_dir / "metadata.json")
            st.success(f"Wrote {csv_path} and {json_path}")
        if col4.button("Contact Sheet PDF"):
            pdf_path = export_contact_sheet(analyses, config.export_dir / "contact_sheet.pdf")
            st.success(f"Wrote {pdf_path}")


def _gallery(analyses, cache: AnalysisCache) -> None:
    columns = st.columns(4)
    for index, item in enumerate(analyses):
        with columns[index % 4]:
            st.video(item.metadata.path)
            st.markdown(f"**{item.score.overall:.1f}** · `{item.metadata.filename}`")
            st.caption(
                f"{item.metadata.duration_seconds:.1f}s · {item.metadata.resolution_label} · "
                f"{item.metadata.orientation} · group {item.duplicate_group or '-'}"
            )
            st.caption(", ".join(item.tags))
            if item.priority_people_matches:
                st.caption("Priority people: " + ", ".join(item.priority_people_matches))
            st.progress(int(item.score.overall))
            with st.expander("Score details"):
                st.write(item.score.as_dict())
            fav = st.checkbox(
                "Favorite", value=item.favorite, key=f"fav-{index}-{item.metadata.path}"
            )
            rej = st.checkbox(
                "Reject", value=item.rejected, key=f"rej-{index}-{item.metadata.path}"
            )
            if fav != item.favorite or rej != item.rejected:
                cache.update_selection(item.metadata.path, fav, rej)
                item.favorite = fav
                item.rejected = rej


def _score_table(analyses) -> None:
    rows = []
    for item in analyses:
        rows.append(
            {
                "Score": item.score.overall,
                "File": item.metadata.filename,
                "Duration": round(item.metadata.duration_seconds, 1),
                "Orientation": item.metadata.orientation,
                "Resolution": item.metadata.resolution_label,
                "Tags": ", ".join(item.tags),
                "Priority People": ", ".join(item.priority_people_matches),
                "People Score": item.score.people_priority,
                "Group": item.duplicate_group or "",
                "Favorite": item.favorite,
                "Rejected": item.rejected,
            }
        )
    st.subheader("Score Table")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _duplicate_compare(analyses) -> None:
    groups = duplicate_summary(analyses)
    if not groups:
        return
    with st.expander("Compare Similar Videos"):
        group_id = st.selectbox("Duplicate group", sorted(groups))
        cols = st.columns(min(4, len(groups[group_id])))
        ranked = sorted(groups[group_id], key=lambda row: row.score.overall, reverse=True)
        for index, item in enumerate(ranked):
            with cols[index % len(cols)]:
                st.video(item.metadata.path)
                st.markdown(f"**{item.score.overall:.1f}** · `{item.metadata.filename}`")


if __name__ == "__main__":
    main()
