# EXP 2 — Transformación log y ablación de features

| Configuración                     |   MAE_test |   RMSE_test |   R2_test |   MAPE_test |
|:----------------------------------|-----------:|------------:|----------:|------------:|
| Sin log + todas las features      |       1237 |        2843 |     0.388 |        30   |
| Con log + todas las features      |       1153 |        2806 |     0.404 |        26.3 |
| Con log + SIN distrito (ablación) |       1237 |        2667 |     0.461 |        29.3 |
| Con log + solo numéricas          |       1337 |        3056 |     0.293 |        32.6 |
