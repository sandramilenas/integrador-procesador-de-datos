from hashlib import sha256
from unittest.mock import call, patch

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from soluciones import (
    ej_1_carga_dataset,
    ej_2_exploracion_dataset,
    ej_3_entrenar_arbol,
    ej_4_evaluacion_rendimiento
)


def test_sol_1():
    X, y = ej_1_carga_dataset()
    
    assert X.shape == (1797, 8, 8)
    assert y.shape == (1797,)
    
    idxs = [154, 1384, 159, 1171, 786, 775, 1251, 169, 361, 1539]
    sample_X = str(X[idxs])
    sample_y = str(y[idxs])
    
    assert sha256(sample_X.encode("ascii")).hexdigest() == "1968799761550fe7f7f816ae869379a5c71a442c3f2e9ce294c5fbe4003953d1"
    assert sha256(sample_y.encode("ascii")).hexdigest() == "b31756ddd427e2d73fc9607eaca826e7a3e9f7bdeea55d29f15efa805c5a50bd"
    
    
@patch("soluciones.plt.show")
@patch("soluciones.plt.title")
@patch("soluciones.plt.imshow")
def test_sol_2(mock_imshow, mock_title, mock_show):
    rng = np.random.default_rng(42)
    X, y = ej_1_carga_dataset()
    
    ej_2_exploracion_dataset(X, y, rng)
    
    expected_idxs = [154, 1384, 159, 1171, 786, 775, 1251, 169, 361, 1539]
    expected = (
        list(
            zip(
                *map(
                    lambda i: (call(X[i], cmap="gray"), call(f"Digito: {y[i]}")),
                    expected_idxs,
                )
            )
        )
    )
    
    for actual_call, expected_call in zip(tuple(mock_imshow.mock_calls), expected[0]):
        np.testing.assert_equal(actual_call.args[0], expected_call.args[0])
        assert actual_call.kwargs["cmap"] == "gray"

    assert tuple(mock_title.mock_calls) == expected[1]
    assert mock_show.mock_calls == [call() for _ in range(len(expected_idxs))]


def test_sol_3():
    X, y = ej_1_carga_dataset()
    
    model = ej_3_entrenar_arbol(X, y)
    
    assert model.get_depth() == 15
    assert model.get_n_leaves() == 145


def test_sol_4():
    X, y = ej_1_carga_dataset()
    model = ej_3_entrenar_arbol(X, y)
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = X_test.reshape(-1, 64)
    
    accuracy = ej_4_evaluacion_rendimiento(model, X_test, y_test)
    
    assert 0.8416666666666667 == pytest.approx(accuracy)
