"""Cut one challenge picture (front left, back right, in the ACE holder) into the views the model sees.

18 views per card picture, in this fixed order:
   0-1   whole front, whole back            FULL_H x FULL_W
   2-9   4 corners of front, 4 of back      CORNER x CORNER, turned so the card corner is top-left
  10-17  4 edges of front, 4 of back        EDGE_DEPTH x EDGE_LENGTH, turned so the outside is on top

Corners and edges are cut at (nearly) the sharpness of the original picture,
because whitening and chipping are only a few pixels wide.
"""
import numpy as np
from PIL import Image

FULL_W, FULL_H = 448, 616          # multiples of 14, close to a card's 63:88 shape
CORNER = 224
EDGE_LENGTH, EDGE_DEPTH = 560, 112
CORNER_SOURCE = 160                # pixels of the original picture that one corner view covers
EDGE_SOURCE_DEPTH = 80             # pixels of the original picture across one edge view
MARGIN = 10                        # pixels kept outside the card so the rim itself is visible
N_VIEWS = 18

# Where the cards sit in a 1412x857 challenge picture (median of the first 100 ACE cards).
REFERENCE_SIZE = (1412, 857)
REFERENCE_BOXES = [(74, 45, 611, 791), (804, 45, 1338, 791)]
CARD_SHAPE = (0.66, 0.78)          # width / height a detected box must have to be trusted

BACKGROUND_TOLERANCE = 40          # colour distance from the background that counts as "card"
MIN_CARD_SHARE = 0.15              # a card must span at least this share of the picture's width
SAME_SIZE = 0.04                   # front and back boxes count as the same size when they differ by no more than this


def to_rgb(image: Image.Image) -> Image.Image:
    """Challenge pictures are RGBA with slightly see-through pixels; lay them on white."""
    if image.mode == "RGBA":
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, image).convert("RGB")
    return image.convert("RGB")


def _runs(present: np.ndarray) -> list[tuple[int, int]]:
    """Start and end of every unbroken stretch of True values."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], present.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def _detect(image: Image.Image) -> list[tuple[int, int, int, int]] | None:
    """Look for the two cards against the background; None when it cannot tell."""
    width, height = image.size
    small_w = 400
    scale = width / small_w
    small = np.asarray(image.resize((small_w, max(1, round(height / scale))), Image.BILINEAR), dtype=np.int16)
    frame = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    is_card = np.abs(small - np.median(frame, axis=0)).sum(axis=2) > BACKGROUND_TOLERANCE

    spans = [s for s in _runs(is_card.mean(axis=0) > 0.3) if s[1] - s[0] >= MIN_CARD_SHARE * small_w]
    if len(spans) < 2:
        return None
    spans = sorted(sorted(spans, key=lambda s: s[0] - s[1])[:2])
    boxes = []
    for left, right in spans:
        rows = _runs(is_card[:, left:right].mean(axis=1) > 0.3)
        if not rows:
            return None
        top, bottom = max(rows, key=lambda r: r[1] - r[0])
        boxes.append((int(left * scale), int(top * scale), int(np.ceil(right * scale)), int(np.ceil(bottom * scale))))
    return boxes


def _card_shaped(box) -> bool:
    """A trading card is 63 x 88 mm, so its box is about 0.72 times as wide as it is tall."""
    width, height = box[2] - box[0], box[3] - box[1]
    return height > 0 and CARD_SHAPE[0] <= width / height <= CARD_SHAPE[1]


def _size_gap(box, other) -> float:
    """How much two boxes differ in size: 0 is the same width and height."""
    return max(abs((box[2] - box[0]) / max(1, other[2] - other[0]) - 1), abs((box[3] - box[1]) / max(1, other[3] - other[1]) - 1))


def find_cards(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Boxes (left, top, right, bottom) of the front card and the back card.

    Front and back are the same card, so the two boxes must have the same size:
      - both sides found with the same size: used as found, whatever the shape
        (small stickers and square cards are cut correctly);
      - the sides disagree (a white card back on the white background shows only
        its print): the card-shaped side nearest to a usual card is trusted and
        the other side gets the same box, moved across by the holder's spacing;
      - nothing trustworthy: the usual place of the cards in the holder.
    """
    width, height = image.size
    sx, sy = width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]
    usual = [tuple(int(round(v)) for v in (l * sx, t * sy, r * sx, b * sy)) for l, t, r, b in REFERENCE_BOXES]
    found = _detect(image)
    if not found:
        return usual
    if _size_gap(found[0], found[1]) <= SAME_SIZE:
        return found
    candidates = [i for i in range(2) if _card_shaped(found[i])]
    if not candidates:
        return usual
    best = min(candidates, key=lambda i: _size_gap(found[i], usual[i]))
    move = (usual[1][0] + usual[1][2] - usual[0][0] - usual[0][2]) // 2   # from the front card across to the back card
    l, t, r, b = found[best]
    other = (l + move, t, r + move, b) if best == 0 else (l - move, t, r - move, b)
    return [found[0], other] if best == 0 else [other, found[1]]


def make_views(image: Image.Image, rng: np.random.Generator | None = None) -> list[Image.Image]:
    """The 18 views. With rng (training only) every card box is nudged by a few pixels."""
    image = to_rgb(image)
    kx, ky = image.size[0] / REFERENCE_SIZE[0], image.size[1] / REFERENCE_SIZE[1]
    corner, depth, margin = CORNER_SOURCE * kx, EDGE_SOURCE_DEPTH * kx, MARGIN * kx
    boxes = find_cards(image)
    if rng is not None:
        boxes = [tuple(v + int(rng.integers(-3, 4)) for v in box) for box in boxes]

    full, corners, edges = [], [], []
    for left, top, right, bottom in boxes:
        l, t, r, b = left - margin, top - margin, right + margin, bottom + margin
        full.append(image.crop((l, t, r, b)).resize((FULL_W, FULL_H), Image.BICUBIC))

        # top-left, top-right, bottom-left, bottom-right; mirrored so the corner always lands top-left
        for x0, y0, flip_x, flip_y in ((l, t, False, False), (r - corner, t, True, False),
                                       (l, b - corner, False, True), (r - corner, b - corner, True, True)):
            view = image.crop((x0, y0, x0 + corner, y0 + corner * ky / kx))
            if flip_x:
                view = view.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_y:
                view = view.transpose(Image.FLIP_TOP_BOTTOM)
            corners.append(view.resize((CORNER, CORNER), Image.BICUBIC))

        # top, bottom, left, right; rotated so the outside of the card is always on top
        for box, turn in (((l, t, r, t + depth), None), ((l, b - depth, r, b), Image.ROTATE_180),
                          ((l, t, l + depth, b), Image.ROTATE_270), ((r - depth, t, r, b), Image.ROTATE_90)):
            view = image.crop(box)
            if turn is not None:
                view = view.transpose(turn)
            edges.append(view.resize((EDGE_LENGTH, EDGE_DEPTH), Image.BICUBIC))
    return full + corners + edges
