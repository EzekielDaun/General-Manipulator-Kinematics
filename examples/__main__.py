import importlib
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m examples <example_name>")
        print("Available examples: foo, bar")
        sys.exit(1)

    name = sys.argv[1]
    try:
        module = importlib.import_module(f"examples.{name}")
    except ModuleNotFoundError as e:
        print(f"Example '{name}' not found.")
        print(e)
        sys.exit(1)

    if hasattr(module, "main"):
        module.main()
    else:
        print(f"Example '{name}' does not define a main() function.")
        sys.exit(1)


if __name__ == "__main__":
    main()
