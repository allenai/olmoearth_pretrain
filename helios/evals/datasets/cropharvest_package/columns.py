class Columns:
    @classmethod
    def tolist(cls) -> list[str]:
        # we don't want to return the magic functions
        return [
            value
            for name, value in vars(cls).items()
            if not (name.startswith("__") or name == "date_columns")
        ]

    @classmethod
    def date_columns(cls) -> list[str]:
        raise NotImplementedError


class RequiredColumns(Columns):
    INDEX = "index"
    IS_CROP = "is_crop"
    LAT = "lat"
    LON = "lon"
    DATASET = "dataset"
    COLLECTION_DATE = "collection_date"
    EXPORT_END_DATE = "export_end_date"
    GEOMETRY = "geometry"
    IS_TEST = "is_test"
    EXTERNALLY_CONTRIBUTED_DATASET = "externally_contributed_dataset"

    @classmethod
    def date_columns(cls) -> list[str]:
        return [cls.COLLECTION_DATE, cls.EXPORT_END_DATE]


class NullableColumns(Columns):
    HARVEST_DATE = "harvest_date"
    PLANTING_DATE = "planting_date"
    LABEL = "label"
    CLASSIFICATION_LABEL = "classification_label"

    @classmethod
    def date_columns(cls) -> list[str]:
        return [cls.HARVEST_DATE, cls.PLANTING_DATE]
