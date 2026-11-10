"""Zero-shot text segmentation on the TerraMesh-Masks eval tiles.

523 tiles of 256 x 256 px at 10 m, each with one binary mask and the short text label it
was drawn for (207 labels; MajorTOM and SSL4EO-S12 groups). OlmoEarth has no text input,
so a tile is scored through a text head: each pixel's embedding against the label's text
embedding, then how well the high-scoring pixels find the mask.

Three steps, each a CLI:

1. ``prep``: TerraMesh's own Sentinel-2 L2A (+ Sentinel-1 RTC) under each mask ->
   ``<inputs>/<file_id>.npz``.
2. ``embed``: a checkpoint's location tokens per pixel, projected through its text head
   when it has one -> ``<emb>/<file_id>.npz`` (``vectors`` [256, 256, D], ``valid``).
3. ``score``: per-tile AP, AUROC, pointing and IoU with bootstrap intervals -> JSON.

Any directory of ``<file_id>.npz`` in that format can be scored, so embeddings read from
elsewhere (e.g. the released archive) go through the same ``score``.
"""
