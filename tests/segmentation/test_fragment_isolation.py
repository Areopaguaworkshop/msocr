from PIL import Image, ImageDraw

from msocr.segmentation.fragment_isolation import isolate_mounted_fragment


def test_mounted_isolation_excludes_frame_label_and_ruler():
    image = Image.new("RGB", (400, 500), "#eee9e5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 10, 395, 490), outline="black", width=14)
    draw.rectangle((145, 110, 285, 400), fill="#a77958")
    draw.rectangle((175, 230, 230, 280), fill="#eee9e5")  # destructive hole
    for y in range(140, 380, 30):
        draw.line((160, y, 270, y), fill="#241b18", width=4)
    draw.rectangle((40, 35, 110, 75), fill="#e3c24e")  # label
    draw.rectangle((330, 80, 350, 430), fill="#b99155")  # ruler

    result = isolate_mounted_fragment(image)

    assert result is not None
    fragment, mask = result
    left, top, right, bottom = fragment.bbox
    assert 120 <= left < 160
    assert 90 <= top < 130
    assert 275 < right <= 310
    assert 390 < bottom <= 425
    assert mask[50, 60] == 0  # label
    assert mask[200, 340] == 0  # ruler
    assert mask[150, 200] == 255  # parchment
    assert mask[250, 200] == 0  # hole remains background
