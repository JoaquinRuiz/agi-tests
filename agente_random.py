"""
Agente aleatorio: la línea base de control.

Sirve para responder a la pregunta incómoda: ¿el modelo lo hace mejor que el azar?
Si un LLM gasta 80 acciones y el azar gasta 80 acciones para llegar al mismo sitio,
no has medido inteligencia, has medido paciencia.

Uso:
    uv run agente_random.py --game ls20 --max-acciones 200
"""

import argparse
import json
import random
import time
from pathlib import Path

import arc_agi
from arcengine import GameAction, GameState
from dotenv import load_dotenv

load_dotenv()  # lee ARC_API_KEY de .env si existe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--max-acciones", type=int, default=200)
    p.add_argument("--semilla", type=int, default=None)
    p.add_argument("--out", default="runs/random.jsonl")
    args = p.parse_args()

    if args.semilla is not None:
        random.seed(args.semilla)

    arc = arc_agi.Arcade()
    env = arc.make(args.game, render_mode="terminal")
    if env is None:
        raise SystemExit("No se pudo crear el entorno. ¿ARC_API_KEY bien puesta?")

    salida = Path(args.out)
    salida.parent.mkdir(parents=True, exist_ok=True)
    log = salida.open("w", encoding="utf-8")

    t0 = time.time()
    niveles = 0
    resultado = "sin_terminar"

    for paso in range(args.max_acciones):
        accion = random.choice(env.action_space)
        datos = {}
        if accion.is_complex():
            datos = {"x": random.randint(0, 63), "y": random.randint(0, 63)}

        obs = env.step(accion, data=datos)

        niveles = getattr(obs, "levels_completed", niveles) if obs else niveles
        log.write(
            json.dumps(
                {
                    "paso": paso,
                    "accion": str(accion),
                    "datos": datos,
                    "estado": str(getattr(obs, "state", None)),
                    "niveles": niveles,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        if obs and obs.state == GameState.WIN:
            resultado = f"ganado_en_{paso}"
            break
        if obs and obs.state == GameState.GAME_OVER:
            env.reset()

    log.close()
    tarjeta = arc.get_scorecard()

    print("\n=== AGENTE ALEATORIO ===")
    print(f"Juego:            {args.game}")
    print(f"Acciones gastadas: {min(paso + 1, args.max_acciones)}")
    print(f"Niveles:          {niveles}")
    print(f"Resultado:        {resultado}")
    print(f"Tiempo:           {time.time() - t0:.1f}s")
    print(f"Scorecard:        {getattr(tarjeta, 'score', tarjeta)}")
    print(f"Log:              {salida}")


if __name__ == "__main__":
    main()
