"""
Agente LLM para ARC-AGI-3, con el harness como variable del experimento.

Este es el script central del vídeo. El mismo modelo, el mismo juego, y un único
parámetro que cambia: si el andamiaje conserva el razonamiento y el historial
entre acciones, o si lo tira a la basura después de cada paso.

    --harness persistente  -> el modelo ve todo lo que ha razonado y probado antes
    --harness amnesico     -> cada acción empieza de cero, solo con el frame actual

Es la réplica en pequeño de lo que midió ARC Prize con GPT-6 Astra: mismo modelo,
62,71% con el harness estándar y 99,9% con el harness que preserva el estado.

Todos los modelos van por OpenRouter, así que solo necesitas UNA clave y puedes
cambiar de proveedor sin tocar el código: basta con cambiar el identificador del modelo.

Uso:
    uv run agente_llm.py --game ls20 --model openai/gpt-6-astra        --harness persistente
    uv run agente_llm.py --game ls20 --model openai/gpt-6-astra        --harness amnesico
    uv run agente_llm.py --game ls20 --model anthropic/claude-fable-5-1 --harness persistente

Requiere OPENROUTER_API_KEY.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import arc_agi
from arcengine import GameAction, GameState
from dotenv import load_dotenv

load_dotenv()  # lee las claves de .env si existe

SISTEMA = """Estás jugando a un videojuego que no has visto nunca. Nadie te va a explicar las reglas.
Tu trabajo es descubrir la mecánica explorando y luego completar los niveles.

En cada turno recibes el estado actual de la pantalla como una rejilla de números,
donde cada número es un color distinto. Debes responder EXCLUSIVAMENTE con un objeto
JSON, sin markdown, sin explicaciones fuera del JSON, con esta forma:

{"accion": "ACTION1", "x": null, "y": null, "hipotesis": "qué creo que hace esta acción",
 "notas": "lo que quiero recordar para los siguientes turnos"}

Las acciones simples no llevan x/y. Las acciones complejas necesitan x e y entre 0 y 63.
El campo "notas" es tu libreta: úsalo para anotar lo que has aprendido de la mecánica."""


# --------------------------------------------------------------------------
# Serialización del frame
# --------------------------------------------------------------------------
def frame_a_texto(obs) -> str:
    """Convierte la observación en texto compacto para el prompt.

    El nombre del campo que contiene la rejilla puede cambiar entre versiones del
    toolkit, así que probamos varios y, si no encontramos ninguno, caemos a repr().
    """
    for campo in ("frame", "frames", "grid", "observation"):
        datos = getattr(obs, campo, None)
        if datos is None:
            continue
        rejilla = datos
        # Algunas versiones devuelven una lista de rejillas (una por capa/tick)
        while isinstance(rejilla, list) and rejilla and isinstance(rejilla[0], list) \
                and rejilla[0] and isinstance(rejilla[0][0], list):
            rejilla = rejilla[-1]
        if isinstance(rejilla, list) and rejilla and isinstance(rejilla[0], list):
            return "\n".join("".join(f"{c:x}" for c in fila) for fila in rejilla)
    return repr(obs)[:4000]


def acciones_disponibles(env) -> str:
    return ", ".join(
        f"{a}{' (necesita x,y)' if a.is_complex() else ''}" for a in env.action_space
    )


# --------------------------------------------------------------------------
# Clientes LLM
# --------------------------------------------------------------------------
class ClienteOpenRouter:
    """Cliente único para todos los modelos, vía OpenRouter.

    OpenRouter habla el mismo protocolo que OpenAI, así que reutilizamos su SDK
    cambiando solo la base_url. Ventaja para este experimento: comparar modelos de
    distintos proveedores sin cambiar ni una línea, solo el identificador.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, modelo: str):
        import os

        from openai import OpenAI

        clave = os.environ.get("OPENROUTER_API_KEY")
        if not clave:
            raise SystemExit("Falta OPENROUTER_API_KEY en el entorno.")

        self.cli = OpenAI(base_url=self.BASE_URL, api_key=clave)
        self.modelo = modelo
        self.json_nativo = True   # se desactiva solo si el modelo no lo soporta
        self.coste = 0.0

    def pedir(self, mensajes: list[dict]) -> tuple[str, int]:
        kwargs = {
            "model": self.modelo,
            "messages": [{"role": "system", "content": SISTEMA}] + mensajes,
            # OpenRouter devuelve el coste real en dólares si se lo pides
            "extra_body": {"usage": {"include": True}},
        }
        if self.json_nativo:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            r = self.cli.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            # No todos los modelos de OpenRouter aceptan json_object.
            # Si falla por eso, reintentamos sin él: el parser ya es tolerante.
            if self.json_nativo and "response_format" in str(e).lower():
                self.json_nativo = False
                kwargs.pop("response_format")
                r = self.cli.chat.completions.create(**kwargs)
            else:
                raise

        uso = r.usage
        self.coste += float(getattr(uso, "cost", 0.0) or 0.0)
        return r.choices[0].message.content, uso.total_tokens


def parsear(texto: str) -> dict:
    t = texto.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        ini, fin = t.find("{"), t.rfind("}")
        if ini >= 0 and fin > ini:
            return json.loads(t[ini : fin + 1])
        raise


# --------------------------------------------------------------------------
# Bucle principal
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--model", required=True,
                   help="identificador de OpenRouter, p.ej. openai/gpt-6-astra")
    p.add_argument(
        "--harness",
        default="persistente",
        choices=["persistente", "amnesico"],
        help="persistente: conserva razonamiento e historial. amnesico: lo descarta cada paso.",
    )
    p.add_argument("--max-acciones", type=int, default=80)
    p.add_argument("--ventana", type=int, default=30, help="turnos de historial que se conservan en modo persistente")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    salida = Path(args.out or f"runs/{args.model.replace(chr(47), chr(45))}_{args.harness}.jsonl")
    salida.parent.mkdir(parents=True, exist_ok=True)
    log = salida.open("w", encoding="utf-8")

    cliente = ClienteOpenRouter(args.model)
    arc = arc_agi.Arcade()
    env = arc.make(args.game, render_mode="terminal")
    if env is None:
        raise SystemExit("No se pudo crear el entorno. ¿ARC_API_KEY bien puesta?")

    historial: list[dict] = []
    notas = ""                 # la libreta que el modelo se pasa a sí mismo
    tokens_total = 0
    niveles = 0
    resultado = "sin_terminar"
    vistas = Counter()          # (hash del frame, acción) -> veces. Detector de bucles
    t0 = time.time()
    obs = None
    paso = 0

    for paso in range(args.max_acciones):
        pantalla = frame_a_texto(obs) if obs is not None else "(pantalla inicial)"

        turno = (
            f"Turno {paso}. Acciones disponibles: {acciones_disponibles(env)}\n"
            f"Niveles completados hasta ahora: {niveles}\n"
        )
        if args.harness == "amnesico":
            # Sin historial y sin notas propias: cada turno redescubre el juego.
            turno += f"Pantalla actual:\n{pantalla}"
            mensajes = [{"role": "user", "content": turno}]
        else:
            # Con historial reciente y con la libreta que él mismo escribió.
            turno += f"Tus notas previas: {notas or '(ninguna todavía)'}\nPantalla actual:\n{pantalla}"
            mensajes = historial[-args.ventana * 2 :] + [{"role": "user", "content": turno}]

        try:
            bruto, tokens = cliente.pedir(mensajes)
        except Exception as e:  # noqa: BLE001
            print(f"[paso {paso}] error del proveedor: {e}")
            break
        tokens_total += tokens

        try:
            decision = parsear(bruto)
        except Exception:  # noqa: BLE001
            print(f"[paso {paso}] respuesta no parseable, la salto")
            continue

        nombre = str(decision.get("accion", "")).upper()
        accion = next((a for a in env.action_space if str(a).upper().endswith(nombre)), None)
        if accion is None:
            print(f"[paso {paso}] acción inválida: {nombre}")
            continue

        datos = {}
        if accion.is_complex():
            datos = {
                "x": int(decision.get("x") or 0) % 64,
                "y": int(decision.get("y") or 0) % 64,
            }

        vistas[(hash(pantalla), nombre, tuple(sorted(datos.items())))] += 1

        obs = env.step(
            accion,
            data=datos,
            reasoning={"thought": str(decision.get("hipotesis", ""))[:500]},
        )

        if args.harness == "persistente":
            notas = str(decision.get("notas", notas))[:2000]
            historial.append({"role": "user", "content": turno})
            historial.append({"role": "assistant", "content": bruto})

        niveles = getattr(obs, "levels_completed", niveles) if obs else niveles

        log.write(
            json.dumps(
                {
                    "paso": paso,
                    "accion": nombre,
                    "datos": datos,
                    "hipotesis": decision.get("hipotesis"),
                    "notas": decision.get("notas"),
                    "estado": str(getattr(obs, "state", None)),
                    "niveles": niveles,
                    "tokens_acumulados": tokens_total,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"[{paso:>3}] {nombre:<10} {datos} niveles={niveles} tokens={tokens_total}")

        if obs and obs.state == GameState.WIN:
            resultado = f"ganado_en_{paso}"
            break
        if obs and obs.state == GameState.GAME_OVER:
            env.reset()

    log.close()
    tarjeta = arc.get_scorecard()
    repetidas = sum(v - 1 for v in vistas.values() if v > 1)
    ejecutadas = max(paso + 1, 1)

    print("\n" + "=" * 52)
    print(f"  MODELO:   {args.model}")
    print(f"  HARNESS:  {args.harness}")
    print("=" * 52)
    print(f"  Acciones ejecutadas:      {ejecutadas}")
    print(f"  Niveles completados:      {niveles}")
    print(f"  Resultado:                {resultado}")
    print(f"  Acciones repetidas:       {repetidas} ({repetidas / ejecutadas:.0%})")
    print(f"  Tokens totales:           {tokens_total}")
    print(f"  Coste real (OpenRouter):  ${cliente.coste:.4f}")
    print(f"  Tiempo:                   {time.time() - t0:.0f}s")
    print(f"  Scorecard:                {getattr(tarjeta, 'score', tarjeta)}")
    print(f"  Log:                      {salida}")
    print("=" * 52)
    print("\n  'Acciones repetidas' = mismo estado de pantalla + misma acción.")
    print("  Es tu métrica de bucle: mide cuánto está redescubriendo lo ya descubierto.\n")


if __name__ == "__main__":
    main()