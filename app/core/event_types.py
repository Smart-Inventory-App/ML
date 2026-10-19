"""Tipos de movimiento de `consumption_events` que cuentan como demanda para predecir.

El uso, la venta y el alquiler sacan producto porque alguien lo necesitó. El vencimiento es una
pérdida y la reposición una entrada: ninguno de los dos mide cuánto se consume.
"""

CONSUMPTION_EVENT_TYPES = ("consumed", "sold", "rented")
