from io import BytesIO

import pytest
from PIL import Image

from src.inference import CLASS_NAMES, inspection_decision, open_image_file


def test_class_list_has_twelve_unique_classes():
    assert len(CLASS_NAMES) == 12
    assert len(set(CLASS_NAMES)) == 12


def test_ok_is_pass():
    assert inspection_decision("ok") == "PASS"


@pytest.mark.parametrize("label", [name for name in CLASS_NAMES if name != "ok"])
def test_defect_classes_are_fail(label):
    assert inspection_decision(label) == "FAIL"


def test_unknown_class_is_not_pass():
    with pytest.raises(ValueError):
        inspection_decision("unknown")


def test_uploaded_image_is_converted_to_rgb():
    source = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    buffer = BytesIO()
    source.save(buffer, format="PNG")
    buffer.seek(0)

    converted = open_image_file(buffer)
    assert converted.mode == "RGB"
    assert converted.size == (8, 8)
