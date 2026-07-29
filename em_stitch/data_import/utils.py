import copy


def pytemca_md_to_temcadb_md(pytemca_md):
    pytemca_md_copy = copy.deepcopy(pytemca_md)
    temcadb_md = {
        "metadata": pytemca_md_copy[0]["metadata"],
        "data": pytemca_md_copy[1]["data"]
    }
    if len(pytemca_md_copy) > 2:
        for k, v in pytemca_md_copy[2].items():
            temcadb_md[k] = v
    return temcadb_md


def temcadb_md_to_pytemca_md(temcadb_md):
    temcadb_md_copy = copy.deepcopy(temcadb_md)
    return [
        {"metadata": temcadb_md_copy.pop("metadata")},
        {"data": temcadb_md_copy.pop("data")},
        temcadb_md_copy
    ]
