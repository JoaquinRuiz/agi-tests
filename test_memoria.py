"""
Test 3: memoria a largo plazo (aprendizaje continuo).

Le enseñamos al modelo un concepto que no puede estar en ningún dataset porque lo
acabas de inventar tú. Luego abrimos una sesión completamente limpia, sin historial,
sin RAG, sin ficheros, y preguntamos lo mismo.

Si en la sesión 1 lo aplica bien y en la sesión 2 no sabe ni de qué le hablas,
el modelo no almacena: solo mantiene en contexto. Que es exactamente lo que dice
el paper de Hendrycks cuando encuentra el déficit en almacenamiento a largo plazo.

Todos los modelos van por OpenRouter: una sola clave, OPENROUTER_API_KEY.

Uso:
    uv run test_memoria.py --model openai/gpt-6-astra
    uv run test_memoria.py --model anthropic/claude-fable-5-1

El concepto se edita en concepto.json. Cámbialo por uno tuyo antes de grabar:
cuanto más personal y absurdo, menos posibilidad de que alguien diga "eso lo sabía ya".
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # lee OPENROUTER_API_KEY de .env si existe


def hacer_cliente(modelo: str):
    """Cliente único vía OpenRouter. Cambiar de proveedor es cambiar el string."""
    import os

    from openai import OpenAI

    clave = os.environ.get("OPENROUTER_API_KEY")
    if not clave:
        raise SystemExit("Falta OPENROUTER_API_KEY en el entorno.")

    cli = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=clave)

    def pedir(mensajes):
        r = cli.chat.completions.create(model=modelo, messages=mensajes)
        return r.choices[0].message.content

    return pedir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="identificador de OpenRouter, p.ej. openai/gpt-6-astra")
    p.add_argument("--concepto", default="concepto.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    c = json.loads(Path(args.concepto).read_text(encoding="utf-8"))
    pedir = hacer_cliente(args.model)

    print("\n" + "=" * 60)
    print(f"  TEST DE MEMORIA — {args.model}")
    print("=" * 60)

    # ---- SESIÓN 1: se lo enseñamos y comprobamos que lo ha entendido -------
    print("\n--- SESIÓN 1: enseñamos el concepto ---\n")
    s1 = [
        {"role": "user", "content": c["ensenanza"]},
    ]
    aprendizaje = pedir(s1)
    print(aprendizaje)

    s1 += [
        {"role": "assistant", "content": aprendizaje},
        {"role": "user", "content": c["pregunta"]},
    ]
    respuesta_1 = pedir(s1)
    print("\n--- SESIÓN 1: aplicación inmediata ---\n")
    print(respuesta_1)

    # ---- SESIÓN 2: sesión nueva, sin absolutamente nada de contexto -------
    print("\n--- SESIÓN 2: sesión limpia, sin historial ---\n")
    respuesta_2 = pedir([{"role": "user", "content": c["pregunta"]}])
    print(respuesta_2)

    # ---- Veredicto: lo juzgas tú, no el script ---------------------------
    print("\n" + "=" * 60)
    print("  VEREDICTO (lo decides tú, a ojo, en cámara)")
    print("=" * 60)
    print("  ¿Aplicó bien el concepto en la sesión 1?      [ SÍ / NO ]")
    print("  ¿Conservaba algo del concepto en la sesión 2? [ SÍ / NO ]")
    print("\n  Si es SÍ / NO, el modelo NO almacena: solo mantiene en contexto.")
    print("  Si es SÍ / SÍ, comprueba que no tengas activada ninguna función")
    print("  de memoria en la cuenta o en el SDK. Y si de verdad no la tienes,")
    print("  enhorabuena, acabas de encontrar el vídeo siguiente.\n")

    destino = Path(args.out or f"runs/memoria_{args.model.replace(chr(47), chr(45))}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                "modelo": args.model,
                "concepto": c["nombre"],
                "sesion_1_aprendizaje": aprendizaje,
                "sesion_1_aplicacion": respuesta_1,
                "sesion_2_limpia": respuesta_2,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Guardado en {destino}\n")


if __name__ == "__main__":
    main()