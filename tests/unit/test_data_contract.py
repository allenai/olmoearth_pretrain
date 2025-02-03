from helios.data_contract import DataContract


def test_all_attrs_have_bands():
    attribute_to_bands = DataContract.attribute_to_bands()
    for attribute_name in DataContract._fields:
        assert attribute_name in attribute_to_bands
