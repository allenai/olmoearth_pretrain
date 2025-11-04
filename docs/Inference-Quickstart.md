## Inference Quickstart

This quickstart shows how to (1) initialize the OlmoEarth model, (2) obtain a satellite
image suitable for input to the model, and (3) compute embeddings from that satellite
image.

## Initializing the Model

First, setup a Python 3.12 environment. You can use your favorite Python package
manager, but here is an example with uv:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/olmoearth_pretrain/
uv sync
```

Now we can use the `olmoearth_pretrain` library to initialize the model in pytorch.
Below, we initialize the OlmoEarth-v1-Base model.

```python
from olmoearth_pretrain.model_loader import ModelID, load_model
model = load_model(ModelID.OLMOEARTH_V1_BASE)
```

## Obtain Satellite Imagery

Here, we obtain one Sentinel-2 image from the ESA Copernicus Browser. If you want to
apply the model on multiple images of a location, like a time series of Sentinel-1 and
Sentinel-2 images, see the
[OlmoEarth embedding](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md).
and [OlmoEarth fine-tuning](https://github.com/allenai/rslearn/blob/master/docs/examples/FinetuneOlmoEarth.md)
guides in rslearn.

To download on image from the Copernicus Browser, follow these steps:

1. Navigate to https://browser.dataspace.copernicus.eu/. Press Login to sign up for an
   account and login.
2. Go to the Search tab at the top-left. Check Sentinel-2, then check L2A. This selects
   Sentinel-2 L2A images, which are the type of Sentinel-2 images that OlmoEarth is
   pre-trained on.
3. Modify the time range if desired. Also, use the area of interest tool at the top
   right to select a spatial area to search over.
4. Then, press Search. We recommend looking through the results to find a less cloudy
   image. You can press Visualize to preview the satellite image in the Browser before
   downloading it. Once you are satisfied, press the download icon next to the image in
   the search results. Once the download is complete, unzip the file.

If you prefer to skip using the browser, you can download and unzip a Sentinel-2 image
of Seattle:

```
wget https://storage.googleapis.com/ai2-rslearn-projects-data/artifacts/example_sentinel2_l2a_scene_of_seattle.zip
unzip example_sentinel2_l2a_scene_of_seattle.zip
```

## Compute Embeddings

Finally, we load the image in Python, normalize it, and apply the model on it to
compute embeddings.

First, we read all of the image bands at 10 m/pixel. We use the B02 band (one of the
10 m/pixel bands) to determine the transform under which to read the remaining bands,
since some are stored at lower resolutions. Note that here we assume that the .SAFE
folder is in the working directory.

```python
import glob
import numpy as np
import rasterio
from olmoearth_pretrain.data.constants import Modality
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

# Get the JP2 filenames that we need to read, in the band order expected by OlmoEarth.
fnames = []
for band_name in Modality.SENTINEL2_L2A.band_order:
    fname = glob.glob(f"*.SAFE/GRANULE/*/IMG_DATA/*/*_{band_name}_*.jp2")[0]
    fnames.append(fname)

# Get the CRS and transform from the first band, which is B02.
with rasterio.open(fnames[0]) as src:
    crs = src.crs
    transform = src.transform
    width = src.width
    height = src.height

# We limit the width/height to 512x512 in case there is limited memory.
width = 512
height = 512

# Now read all of the bands.
image = np.zeros((len(fnames), height, width), dtype=np.int32)
for band_idx, fname in enumerate(fnames):
    with rasterio.open(fname) as src:
        with rasterio.vrt.WarpedVRT(
            src,
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
        ) as vrt:
            image[band_idx, :, :] = vrt.read(1)

# Rearrange to BHWTC.
image = image.transpose(1, 2, 0)[None, :, :, None, :]
```

Next, we normalize the image:

```python
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

normalizer = Normalizer(Strategy.COMPUTED)
image = normalizer.normalize(Modality.SENTINEL2_L2A, image)
```

Now we can apply the model on the image. We recommend applying it on inputs between
1x1 and 128x128 in size. The patch size can be set between 1 and 8; smaller patch sizes
generally perform better, but require more GPU time.

```python
import torch
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

device = torch.device("cuda")
model.to(device)

# Run the model on the topleft 64x64 of the image.
sample = MaskedOlmoEarthSample(
    sentinel2_l2a=torch.tensor(image[:, 0:64, 0:64, :, :], dtype=torch.float32, device=device),
    # The mask shape is BHWTS, where S is the number of band sets (3 for Sentinel-2).
    sentinel2_l2a_mask=torch.ones((1, 64, 64, 1, 3), dtype=torch.float32, device=device) * MaskValue.ONLINE_ENCODER.value,
    # The timestamps is (day of month 1-31, month 0-11, year).
    # The values here correspond to the date of our sample Sentinel-2 image of Seattle
    # (2025-08-22).
    timestamps=torch.tensor([22, 7, 2025], device=device)[None, None, :],
)
tokens_and_masks = model.encoder(
    sample, fast_pass=True, patch_size=4,
)["tokens_and_masks"]
# Get the Sentinel-2 features.
modality_features = tokens_and_masks.sentinel2_l2a
# Pool the features over the timestep and band set dimensions so we end up with a BHWC
# feature map.
pooled = modality_features.mean(dim=[3, 4])
```
