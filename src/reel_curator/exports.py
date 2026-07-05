from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from reel_curator.models import VideoAnalysis


def export_videos(
    analyses: list[VideoAnalysis], output_dir: Path, mode: str, top_n: int = 25
) -> Path:
    selected = select_for_export(analyses, mode, top_n)
    target_dir = output_dir / mode.replace(" ", "_").lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        source = Path(item.metadata.path)
        destination = target_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    return target_dir


def select_for_export(
    analyses: list[VideoAnalysis], mode: str, top_n: int = 25
) -> list[VideoAnalysis]:
    ranked = sorted(
        [item for item in analyses if not item.rejected],
        key=lambda item: item.score.overall,
        reverse=True,
    )
    if mode == "favorites":
        return [item for item in ranked if item.favorite]
    if mode == "top_n":
        return ranked[:top_n]
    if mode == "duplicate_representatives":
        return [
            item
            for item in ranked
            if item.duplicate_representative or not item.duplicate_group
        ]
    raise ValueError(f"Unknown export mode: {mode}")


def export_csv(analyses: list[VideoAnalysis], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.as_dict() for item in analyses]
    fieldnames = sorted({key for row in rows for key in row.keys()} - {"score"})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames}
            flat["tags"] = ", ".join(row.get("tags", []))
            flat["vision_labels"] = ", ".join(row.get("vision_labels", []))
            writer.writerow(flat)
    return output_path


def export_json(analyses: list[VideoAnalysis], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([item.as_dict() for item in analyses], handle, indent=2)
    return output_path


def export_contact_sheet(analyses: list[VideoAnalysis], output_path: Path, limit: int = 80) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_size = landscape(letter)
    pdf = canvas.Canvas(str(output_path), pagesize=page_size)
    width, height = page_size
    margin = 0.35 * inch
    card_w = (width - 2 * margin) / 4
    card_h = 1.7 * inch
    thumb_h = 1.05 * inch
    x = margin
    y = height - margin - card_h

    ranked = sorted(analyses, key=lambda row: row.score.overall, reverse=True)[:limit]
    for index, item in enumerate(ranked):
        if index and index % 16 == 0:
            pdf.showPage()
            x = margin
            y = height - margin - card_h

        pdf.setStrokeColor(colors.lightgrey)
        pdf.rect(x, y, card_w - 0.08 * inch, card_h, stroke=1, fill=0)
        thumb_path = Path(item.thumbnail_path)
        if thumb_path.exists():
            _draw_image(
                pdf,
                thumb_path,
                x + 0.05 * inch,
                y + card_h - thumb_h - 0.05 * inch,
                card_w - 0.18 * inch,
                thumb_h,
            )
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(
            x + 0.06 * inch,
            y + 0.43 * inch,
            f"{item.score.overall:.1f}  {item.metadata.filename[:28]}",
        )
        pdf.setFont("Helvetica", 7)
        tags = ", ".join(item.tags[:4])
        pdf.drawString(
            x + 0.06 * inch,
            y + 0.25 * inch,
            f"{item.metadata.duration_seconds:.1f}s  {item.metadata.orientation}  {tags[:34]}",
        )
        pdf.drawString(x + 0.06 * inch, y + 0.09 * inch, f"Group: {item.duplicate_group or '-'}")

        x += card_w
        if x + card_w > width - margin / 2:
            x = margin
            y -= card_h + 0.15 * inch
    pdf.save()
    return output_path


def _draw_image(pdf: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float) -> None:
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    draw_w = iw * scale
    draw_h = ih * scale
    pdf.drawImage(
        str(path),
        x + (w - draw_w) / 2,
        y + (h - draw_h) / 2,
        draw_w,
        draw_h,
        preserveAspectRatio=True,
    )
