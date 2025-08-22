import numpy as np


def err_weighted_mean(file_path, value_col=0, error_col=1, result_length=4):
    """
    Berechnet den gewichteten Mittelwert und den zugehörigen Fehler eines Wertearrays unter Berücksichtigung individueller Fehlerwerte.

    Parameter:
    - file_path (str): Der relative Pfad zu der zu mittelnden Datei.
    - value_col (int, optional): Index der Spalte mit den Werten (Standard: 0).
    - error_col (int, optional): Index der Spalte mit den Fehlern (Standard: 1).
    - result_length (int, optional): Anzahl der Nachkommastellen für die Ergebnisse (Standard: 4).

    Rückgabewert:
    - mean_val (float): Der berechnete gewichtete Mittelwert.
    - err_mean_val (float): Der Fehler des Mittelwerts.
    """
    # Datei einlesen
    data = np.loadtxt(file_path)

    # Werte und Fehler aus den angegebenen Spalten extrahieren
    val = data[:, value_col]  # Die Werte
    err_val = data[:, error_col]  # Die Fehler

    # Berechnung des Mittels und des Fehlers
    mean = mean_calc(val, err_val, 'data weighting')
    err_mean = mean_calc(val, err_val, 'error')

    # Rundung auf result_length Nachkommastellen
    mean_round = round(float(mean), result_length)
    err_mean_round = round(float(err_mean), result_length)

    return mean_round, err_mean_round


def mean_calc(z_input, err_input, goal='data weighting'):
    """
    Berechnet den gewichteten Mittelwert eines Wertearrays unter Berücksichtigung individueller Fehlerwerte.

    Parameter:
    - z_input (array_like): Array mit den zu gewichtenden Werten.
    - err_input (array_like): Array mit den Fehlern, die den Werten in z_input zugeordnet sind.
    - goal (str, optional): Bestimmt den Berechnungsmodus.
        - 'data weighting': Berechnet den gewichteten Mittelwert der Daten.
        - 'error': Berechnet den Fehler des Mittelwerts.
        Standardwert: 'data weighting'

    Rückgabewert:
    - mean_val (float): Der berechnete gewichtete Mittelwert oder der Fehler des Mittelwerts.
    """
    # Use vectorized numpy operations for performance and numerical stability
    z = np.asarray(z_input, dtype=float)
    err = np.asarray(err_input, dtype=float)

    if np.any(err == 0):
        # avoid division by zero; treat zero-errors as very large weight (effectively ignore)
        # Here we treat zero error as very small uncertainty -> very large weight; user data should avoid exact zero.
        err = np.where(err == 0, np.finfo(float).tiny, err)

    weights = (1.0 / err) ** 2

    if goal == 'data weighting':
        return np.sum(z * weights) / np.sum(weights)
    elif goal == 'error':
        return np.sqrt(1.0 / np.sum(weights))
    else:
        raise ValueError("Ungültiges Ziel für mean_calc. Nutze 'data weighting' oder 'error'.")