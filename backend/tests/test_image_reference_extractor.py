from backend.context.image_reference_extractor import (
    extract_png_image_references_from_text,
    normalize_image_file_name,
    normalize_raw_image_reference,
)


def test_normalize_raw_image_reference_strips_wrappers_and_backslashes():
    assert (
        normalize_raw_image_reference('  "images\\GUID-AB12-low.png"  ')
        == "images/GUID-AB12-low.png"
    )
    assert normalize_raw_image_reference("GUID-AB12-low.png,") == "GUID-AB12-low.png"
    assert normalize_raw_image_reference(None) == ""


def test_normalize_image_file_name_returns_basename():
    assert (
        normalize_image_file_name("figures/sub/GUID-AB12-CD34-low.png")
        == "GUID-AB12-CD34-low.png"
    )


def test_extract_png_references_finds_guid_images_with_context():
    text = (
        "Remove the breather cap as shown. GUID-1A2B-3C4D-low.png "
        "Then torque the bolts to spec."
    )
    refs = extract_png_image_references_from_text(text)

    assert len(refs) == 1
    ref = refs[0]
    assert ref["fileName"] == "GUID-1A2B-3C4D-low.png"
    assert ref["imageIndexInChunk"] == 1
    assert "breather cap" in ref["textBeforeImage"]
    assert "torque the bolts" in ref["textAfterImage"]


def test_extract_png_references_deduplicates_and_ignores_non_matches():
    text = "GUID-XYZ9-low.png and again GUID-XYZ9-low.png plus photo.jpg"
    refs = extract_png_image_references_from_text(text, include_nearby_text=False)

    assert len(refs) == 1
    assert refs[0]["fileName"] == "GUID-XYZ9-low.png"
    assert "nearbyText" not in refs[0]


def test_extract_png_references_empty_text():
    assert extract_png_image_references_from_text(None) == []
    assert extract_png_image_references_from_text("") == []
