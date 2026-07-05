from io import BytesIO

from PIL import Image

from reel_curator.face_priority import normalized_reference_image


def test_normalized_reference_image_applies_exif_orientation(tmp_path) -> None:
    image = Image.new("RGB", (20, 40), color="red")
    exif = image.getexif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    path = tmp_path / "rotated.jpg"
    path.write_bytes(buffer.getvalue())

    normalized = normalized_reference_image(path)

    assert normalized is not None
    with Image.open(BytesIO(normalized)) as normalized_image:
        assert normalized_image.size == (40, 20)
