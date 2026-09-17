# agi-tests — las pruebas del vídeo "¿Es GPT-6 AGI?"

Tres scripts para responder con datos a la pregunta del título, en lugar de con titulares,
más una rúbrica para convertir esos datos en una nota repetible.

📺 **Vídeo:** [¿Es GPT-6 AGI?](https://www.youtube.com/watch?v=n5QAvo1QeOI)

## Montaje

```bash
git clone <este-repo> && cd agi-tests
uv sync                       # instala arc-agi, openai, pyyaml y python-dotenv
cp .env.example .env          # y rellena tus claves dentro
```

`.env` guarda tus claves y **no se sube** (está en `.gitignore`); los scripts lo cargan
solos con `python-dotenv`. Las dos claves:

- `OPENROUTER_API_KEY` — la única clave de modelo que necesitas.
- `ARC_API_KEY` — opcional: sin clave se usa una anónima.

Si prefieres no usar `.env`, expórtalas a mano con `export OPENROUTER_API_KEY="..."`.

Todos los modelos van por **OpenRouter**, que habla el mismo protocolo que la API de
OpenAI. Una sola clave, un solo cliente en el código, y cambiar de proveedor es cambiar
un string: `openai/gpt-6-astra`, `anthropic/claude-fable-5-1`, y lo que salga mañana.

## Test 1 — Inteligencia fluida (MEDIR-01 / MEDIR-02)

Primero juega tú en el navegador, cronómetro en mano, sin leer nada.

Luego la línea base del azar, que es el control que casi nadie pone:

```bash
uv run agente_random.py --game ls20 --max-acciones 200 --semilla 42
```

Y después el modelo:

```bash
uv run agente_llm.py --game ls20 --model openai/gpt-6-astra --harness persistente
```

## Test 2 — El andamiaje (MEDIR-04)

El experimento entero es cambiar una palabra. Mismo modelo, mismo juego, misma semilla mental:

```bash
uv run agente_llm.py --game ls20 --model openai/gpt-6-astra --harness persistente
uv run agente_llm.py --game ls20 --model openai/gpt-6-astra --harness amnesico
```

En modo `amnesico` cada turno llega sin historial y sin las notas que el propio modelo
escribió antes, que es en esencia lo que hacía el harness estándar de ARC-AGI-3:
descartar el razonamiento después de cada acción y truncar los movimientos antiguos.

Compara tres cifras entre las dos ejecuciones: niveles completados, tokens totales y el
porcentaje de acciones repetidas. Esa última es tu métrica de bucle, y es la que mejor se
ve en cámara.

## Test 3 — Memoria a largo plazo (MEDIR-05 / MEDIR-06)

```bash
uv run test_memoria.py --model openai/gpt-6-astra
uv run test_memoria.py --model anthropic/claude-fable-5-1
```

**Edita `concepto.json` antes de grabar.** El que viene de ejemplo sirve para probar el
script, pero para el vídeo usa una convención tuya de verdad: cuanto más personal, menos
margen para el comentario de "eso ya lo sabía de antes".

Asegúrate de que la memoria de la aplicación está desactivada y grábalo en pantalla, o la
mitad de los comentarios va a ir por ahí.

## La rúbrica — de mediciones a nota

Los tres tests te dan números en crudo. `rubrica.py` los convierte en un veredicto
repetible leyendo dos ficheros YAML (con `yaml.safe_load`):

- `rubrica.yaml` — los cinco criterios de AGI, cada uno con su umbral comprobable y su
  fuente (ARC-AGI-3, METR, tus propias ejecuciones). No es opinión, es un contrato: si no
  te gusta un umbral, cámbialo y vuelve a puntuar.
- `resultados.yaml` — donde vuelcas **tus** mediciones por modelo. `null` es "no medido",
  y un hueco no cuenta ni como cero ni como aprobado.

```bash
uv run rubrica.py --sistema gpt-6-astra
uv run rubrica.py --sistema gpt-6-astra --agregacion media
```

Los criterios son inteligencia fluida, independencia del andamiaje, memoria a largo plazo,
horizonte temporal al 80% y coherencia (que ningún dominio tenga un cero). Por defecto la
nota se agrega con el **mínimo**, no con la media: una cadena es tan fuerte como su eslabón
más débil, y con media un 0% en memoria te lo tapa un sobresaliente en otra cosa. Ejecútalo
con `--agregacion media` (o cambia el campo en `rubrica.yaml`) para ver cuánto se mueve el
veredicto solo con eso.

Ni el agente ni el test de memoria tocan estos ficheros: la rúbrica es una capa aparte que
solo lee lo que tú anotas. Por eso `pyyaml` es dependencia del proyecto aunque los otros
scripts no la usen.

## Alcance

Los scripts están ejecutados contra la API real del toolkit (`arc_agi.Arcade`, `env.make`,
`env.step`, `arc.get_scorecard`). Lo único que te tocará cambiar son **los identificadores
de modelo**: ponlos como aparezcan en el catálogo de OpenRouter, que es también donde ves
el precio por millón de tokens antes de lanzar.

Y el aviso que conviene decir también en el vídeo: esto no es una evaluación oficial. La
evaluación oficial de ARC-AGI-3 sobre GPT-6 Astra costó veintiséis mil dólares en el run del
harness estándar. Lo que haces aquí es una réplica casera del *efecto*, no del número.

## Licencia

Publicado bajo la licencia [MIT](LICENSE).

## About the author

**Joaquín Ruiz** — [jokiruiz.com](https://jokiruiz.com) · [youtube.com/@jokioki](https://youtube.com/@jokioki)

Autor de:

- 📗 [*Del vibe coding al Spec-Driven Development*](https://amzn.eu/d/02csLpKC)
- 📙 [*El motor de la Inteligencia Artificial*](https://amzn.eu/d/083CTN3U)
- 📘 [*Programar con Inteligencia Artificial*](https://amzn.eu/d/eK4f73N)
- 📙 [*Explora la Inteligencia Artificial*](https://amzn.eu/d/dSwYhue)
