//------------------------------------------------------------------------------------
// Evaluates Togo cropland maps
// Author: Ivan Zvonkov (ivan.zvonkov@gmail.com)
//------------------------------------------------------------------------------------

var roi = ee
    .FeatureCollection('FAO/GAUL/2015/level2')
    .filter("ADM0_NAME=='Togo'");
Map.centerObject(roi, 7);

function evaluate(mapName, map) {
    var testSet = map.sampleRegions(testPoints, ['is_crop'], 10);
    var errMatrix = testSet.errorMatrix('is_crop', 'map_crop');
    print(
        mapName,
        'Overall Accuracy: ' + errMatrix.accuracy().getInfo(),
        "User's Accuracy: " + errMatrix.consumersAccuracy().getInfo()[0][1],
        "Producer's Accuracy: " + errMatrix.producersAccuracy().getInfo()[1][0],
        'F1 Score: ' + errMatrix.fscore().getInfo()[1]
    );
}

var points = ee.FeatureCollection('users/izvonkov/Togo/points_2019');
var testPoints = points.filter(ee.Filter.eq('subset', 'testing'));

// Load OlmoEarth embeddings for region of interest
var classVis = { min: 0, max: 1.0, palette: ['yellow', 'green'] };
var OlmoEarthCropland = ee
    .Image('projects/ai2-ivan/assets/Togo_v2_cropland_preds')
    .gt(0.5)
    .rename('map_crop');
Map.addLayer(
    OlmoEarthCropland,
    classVis,
    'OlmoEarth Based Cropland',
    true,
    0.5
);

var WorldCover = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi);
var WorldCoverCropland = WorldCover.eq(40).rename('map_crop');
var GLAD = ee.ImageCollection('users/potapovpeter/Global_cropland_2019');
var GLADCropland = GLAD.mosaic().clip(roi).gt(0.5).rename('map_crop');
Map.addLayer(WorldCoverCropland, classVis, 'WorldCover Cropland', true, 0.5);
Map.addLayer(GLADCropland, classVis, 'GLAD Cropland', true, 0.5);

Map.setOptions('satellite');

evaluate('WorldCover', WorldCoverCropland);
evaluate('GLAD', GLADCropland);
evaluate('OlmoEarth', OlmoEarthCropland);
