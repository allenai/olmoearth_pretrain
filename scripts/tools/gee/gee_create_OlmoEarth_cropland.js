//------------------------------------------------------------------------------------
// Generates Togo cropland map from embeddings
// Author: Ivan Zvonkov (ivan.zvonkov@gmail.com)
//------------------------------------------------------------------------------------

// 1. Load embeddings for region of interest
var roi = ee
    .FeatureCollection('FAO/GAUL/2015/level2')
    .filter("ADM0_NAME=='Togo'");
Map.centerObject(roi, 7);

var prefix = ee.String(
    'gs://ai2-ivan-helios-output-data/Togo_v2_nano_2019-03-01_2020-03-01/'
);
var uris = ee.List.sequence(0, 656).map(function (i) {
    return prefix.cat(ee.Number(i).int()).cat(ee.String('.tif'));
});
var images = uris.map(function (uri) {
    return ee.Image.loadGeoTIFF(uri);
});
var embeddings = ee.ImageCollection.fromImages(images).mosaic();

// User memory limit exceeded.
// Map.addLayer(embeddings, {min:0, max: 0.2}, 'Embeddings');

// 2. Load Togo points
var points = ee.FeatureCollection('users/izvonkov/Togo/points_2019');
var trainingPoints = points.filter(ee.Filter.eq('subset', 'training'));

// 3. Create training dataset (training points + embeddings)
var trainingSet = embeddings.sampleRegions(trainingPoints, ['is_crop'], 10);

// 4. Train a classifier (100 trees)
var model = ee.Classifier.smileRandomForest(100).setOutputMode('probability');
var trainedModel = model.train(trainingSet, 'is_crop', embeddings.bandNames());

// 5. Classify embeddings using trained model
var croplandPreds = embeddings.classify(trainedModel).clip(roi);
var croplandMap = croplandPreds.gte(0.7).rename('map_crop');

// 6. Display cropland maps (memory exceeded)
//var classVis = {min: 0, max: 1.0, palette: ['yellow', 'green']}
//Map.addLayer(croplandMap, classVis, 'Presto Embeddings Based Cropland');

Export.image.toAsset({
    image: croplandPreds,
    description: 'croplandPreds',
    assetId: 'Togo_v2_cropland_preds',
    crs: 'EPSG:25231',
    maxPixels: 2615797346,
    scale: 10,
});
