import importlib.util
import types
import sys
import os
import numpy as np
import pytest


def load_cluster_processor():
    base = os.path.join(os.path.dirname(__file__), os.pardir, 'src')
    vfscript = types.ModuleType('vfscript')
    utils = types.ModuleType('vfscript.utils')

    spec_uc = importlib.util.spec_from_file_location(
        'vfscript.utils.utilidades_clustering',
        os.path.join(base, 'vfscript', 'utils', 'utilidades_clustering.py')
    )
    uc = importlib.util.module_from_spec(spec_uc)
    spec_uc.loader.exec_module(uc)

    utils.utilidades_clustering = uc
    vfscript.utils = utils
    sys.modules['vfscript'] = vfscript
    sys.modules['vfscript.utils'] = utils
    sys.modules['vfscript.utils.utilidades_clustering'] = uc

    spec_cp = importlib.util.spec_from_file_location(
        'vfscript.cluster_processing.cluster_processor',
        os.path.join(base, 'vfscript', 'cluster_processing', 'cluster_processor.py')
    )
    cp = importlib.util.module_from_spec(spec_cp)
    spec_cp.loader.exec_module(cp)
    return cp


def test_compute_dispersion_basic():
    cp = load_cluster_processor()
    coords = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0]])
    labels = np.array([0, 0, 0, 1])
    dispersion = cp.compute_dispersion(coords, labels)
    assert pytest.approx(dispersion[0]) == 0.9428090415820634
    assert dispersion[1] == pytest.approx(0.0)


def test_silhouette_mean_basic():
    cp = load_cluster_processor()
    coords = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]])
    labels = np.array([0, 0, 1])
    value = cp.silhouette_mean(coords, labels)
    assert value == pytest.approx(0.5962962962962962)

