# 🎤 Guion de la presentación (5 minutos) — cómo exponer y repartirnos

Proyecto: **Predicción de precios de alquiler en Lima con Machine Learning**
Curso: Inteligencia Artificial · Dr. Vicente Machaca Arceda · ULaSalle

> Este documento dice, para cada integrante, **qué hacer con las manos y qué decir
> con la boca**, cronometrado para que no pasemos de 5 minutos.

---

## 👥 Reparto de roles (completar con nombres)

| Bloque | Quién habla | Nombre |
|---|---|---|
| ① Problema | Integrante 1 | … |
| ② Datos | Integrante 2 | … |
| ③ Pipeline y técnica | Integrante 3 | … |
| ④ Experimentos | Integrante 4 | … |
| ⑤ Demo en vivo | Integrante 5 (Jerson, comparte pantalla) | … |
| ⑥ Cierre | Integrante 1 (o quien prefiera) | … |
| Preguntas del profe | Todos, según el tema | — |

> Somos 5 y cada uno toma un bloque; el ⑥ (cierre, muy corto) lo hace quien ya
> habló en el ①. La exposición es solo hablada.

---

## 🔧 ANTES DE EMPEZAR (5 minutos antes del turno) — lo hace quien comparte pantalla

1. Abrir PowerShell y dejar la app **ya corriendo**:
   ```powershell
   cd C:\Users\Jerson\Desktop\mis-proyectos\ia-precios-lima
   .venv\Scripts\activate
   streamlit run app/app.py --server.headless true
   ```
   *(el `--server.headless true` evita que pida email y arranca limpio)*
2. Se abre en `http://localhost:8501`. Dejar en **Miraflores, 90 m², 2 dorm, 2 baños**. No hacer clic todavía.
3. Hacer **un clic de prueba** en "Estimar alquiler" y recargar la página → así el modelo queda cacheado y el primer clic en vivo es instantáneo.
4. Tener pestañas abiertas: (a) la app, (b) el repo en GitHub, (c) el README (por si preguntan el diagrama o las tablas).

---

## 🗣️ GUION EN VIVO

### ① Problema — 30 seg
> "Buenas, nuestro proyecto predice el precio de alquiler de inmuebles en Lima.
> El problema real es que el mercado publica precios en soles y en dólares, sin
> estándar, y nadie tiene una referencia objetiva de cuánto debería costar un
> departamento. Lo resolvimos con Machine Learning."

### ② Datos — 40 seg
> "Usamos datos **reales** de Properati, publicados en Zenodo con DOI y licencia
> Creative Commons. Son 1,079 avisos reales de abril de 2023. Como venían con
> monedas mezcladas, los unificamos a soles, filtramos solo Lima y quedamos con
> 867 registros limpios."

### ③ Pipeline y técnica — 60 seg
> "El pipeline es: descarga → limpieza → análisis exploratorio → preprocesamiento
> con un ColumnTransformer → modelo → evaluación. La técnica es **regresión con
> Scikit-learn**, porque el precio es un número continuo. Un detalle clave: el
> análisis exploratorio mostró que el precio está muy sesgado, así que entrenamos
> sobre el **logaritmo** del precio, y eso mejora las métricas."

### ④ Experimentos — 90 seg *(EL BLOQUE MÁS IMPORTANTE)*
> "Hicimos tres experimentos.
> **Uno:** comparamos cinco modelos. Los de árboles ganaron; los lineales dieron
> R² negativo porque no manejan el sesgo. Ganó Random Forest.
> **Dos:** probamos quitar la variable 'distrito' y, sorpresa, el modelo *mejoró*.
> Con solo 693 filas, codificar 23 distritos causa sobreajuste. Ese hallazgo
> muestra que entendemos *por qué* funciona el modelo, no solo que funciona.
> **Tres:** ajustamos hiperparámetros con GridSearchCV y no mejoró: el techo lo
> pone la cantidad de datos, no el ajuste fino."

### ⑤ DEMO EN VIVO — 75 seg *(cambiar a la app)*
> "Y esto es la aplicación funcional."
>
> **Pasos:**
> 1. Con Miraflores / 90 m² → clic en **"Estimar alquiler"** → sale ~**S/ 3,182**.
> 2. "Un depa así en Miraflores: ~3,200 soles. Veamos uno más económico."
> 3. Cambiar distrito a **Comas** y área a **60** → clic → baja a ~**S/ 1,300**.
> 4. "El modelo respeta la lógica del mercado: misma app, otro distrito, otro
>    precio. Además da un rango de confianza y el equivalente en dólares."

### ⑥ Cierre — 25 seg
> "En conclusión: logramos un MAE de unos 1,150 soles y un R² de 0.40. Es modesto
> pero **honesto**: el alquiler depende de cosas que el dataset no captura, como el
> piso o si está amoblado. La lección es que en datos reales el preprocesamiento y
> el tamaño de muestra pesan más que el algoritmo. Todo está en el repo,
> reproducible con un solo comando. Gracias."

---

## 🛡️ SI EL PROFE PREGUNTA (respuestas listas)

- **"¿Por qué el R² es bajo?"** → "Features limitadas y solo 867 filas. Como
  trabajo futuro proponemos agrupar distritos en zonas socioeconómicas y añadir
  piso / amoblado / cochera."
- **"¿Por qué Random Forest y no otro?"** → "Ganó por validación cruzada de 5
  folds, que es el criterio honesto; no elegimos por el resultado del test."
- **"¿Los datos son reales?"** → "Sí, Properati vía Zenodo, DOI
  10.5281/zenodo.7846211, licencia CC-BY. El CSV está en el repo."
- **"¿Es reproducible?"** → "Sí: `python run_pipeline.py` regenera datos, modelo
  y experimentos desde cero."
- **"¿Hubo fuga de datos (data leakage)?"** → "No: la imputación y el escalado se
  ajustan solo con el train, dentro de un ColumnTransformer, y evaluamos en un test
  que el modelo nunca vio."

---

## ✅ CHECKLIST 30 SEG ANTES
- [ ] App corriendo y respondiendo (clic de prueba + recarga hecho).
- [ ] Repo abierto en una pestaña.
- [ ] Cada uno sabe su bloque.
- [ ] Cronómetro / alguien controlando los 5 minutos.

---

## 🔗 Enlaces del proyecto
- **Repositorio:** https://github.com/JersonCh1/ia-precios-alquiler-lima
- **README:** https://github.com/JersonCh1/ia-precios-alquiler-lima#readme
- **Enunciado y rúbrica:** `docs/ENUNCIADO.md`
- **Checklist de rúbrica:** `docs/RUBRICA_CHECKLIST.md`
