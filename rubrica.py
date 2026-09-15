"""
rubrica.py — puntúa un sistema contra la rúbrica AGI.

Lee los criterios de rubrica.yaml y las mediciones de resultados.yaml, y dicta
veredicto. Lo interesante es que puedes ejecutarlo dos veces, cambiando solo el
campo 'agregacion', y ver cómo el mismo sistema aprueba o suspende según cómo sumes.

Uso:
    uv run rubrica.py --sistema gpt-6-astra
    uv run rubrica.py --sistema gpt-6-astra --agregacion media
"""

import argparse
import operator
from pathlib import Path

import yaml

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt,
       "<": operator.lt, "==": operator.eq}

VERDE, ROJO, GRIS, FIN = "\033[92m", "\033[91m", "\033[90m", "\033[0m"


def evaluar(criterio: str, valor) -> bool | None:
    """Aplica un criterio del tipo '>= 1.0' o '== true' a un valor medido."""
    if valor is None:
        return None
    op_txt, _, esperado_txt = criterio.strip().partition(" ")
    op = OPS.get(op_txt)
    if op is None:
        raise ValueError(f"Operador no soportado: {op_txt}")
    esperado = esperado_txt.strip()
    if esperado in ("true", "false"):
        return op(bool(valor), esperado == "true")
    return op(float(valor), float(esperado))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sistema", required=True)
    p.add_argument("--rubrica", default="rubrica.yaml")
    p.add_argument("--resultados", default="resultados.yaml")
    p.add_argument("--agregacion", default=None, choices=["minimo", "media"])
    args = p.parse_args()

    rubrica = yaml.safe_load(Path(args.rubrica).read_text(encoding="utf-8"))
    todos = yaml.safe_load(Path(args.resultados).read_text(encoding="utf-8"))
    medido = todos.get(args.sistema)
    if medido is None:
        raise SystemExit(
            f"No hay mediciones para '{args.sistema}' en {args.resultados}. "
            f"Disponibles: {', '.join(todos)}"
        )

    agregacion = args.agregacion or rubrica.get("agregacion", "minimo")

    print("\n" + "=" * 62)
    print(f"  RÚBRICA AGI v{rubrica['version']}  ·  sistema: {args.sistema}")
    print(f"  agregación: {agregacion}")
    print("=" * 62)

    puntos, sin_medir = [], []
    for t in rubrica["tests"]:
        valor = medido.get(t["id"])
        resultado = evaluar(t["criterio"], valor)

        if resultado is None:
            marca, color = "SIN MEDIR", GRIS
            sin_medir.append(t["nombre"])
        else:
            marca = "PASA" if resultado else "FALLA"
            color = VERDE if resultado else ROJO
            puntos.append(1.0 if resultado else 0.0)

        print(f"\n  {color}[{marca:^9}]{FIN} {t['nombre']}")
        print(f"             criterio: {t['criterio']}   medido: {valor}")
        print(f"             fuente:   {t['fuente']}")

    print("\n" + "-" * 62)

    if not puntos:
        raise SystemExit("  No hay ningún test medido. Rellena resultados.yaml.\n")

    nota = min(puntos) if agregacion == "minimo" else sum(puntos) / len(puntos)
    es_agi = nota >= 1.0

    print(f"  Tests evaluados:  {len(puntos)} de {len(rubrica['tests'])}")
    print(f"  Nota ({agregacion}): {nota:.2f}")
    print(f"  VEREDICTO:        {(VERDE + 'ES AGI' if es_agi else ROJO + 'NO ES AGI')}{FIN}")

    if sin_medir:
        print(f"\n  {GRIS}Sin medir: {', '.join(sin_medir)}.")
        print(f"  Un hueco no es un aprobado. Comparar un número con un hueco{FIN}")
        print(f"  {GRIS}no es comparar, es rellenar con fe.{FIN}")

    if agregacion == "media" and not es_agi and min(puntos) == 0:
        print(f"\n  {GRIS}Aviso: con agregación 'media' un cero estructural queda")
        print(f"  compensado por el resto. Prueba --agregacion minimo.{FIN}")

    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()