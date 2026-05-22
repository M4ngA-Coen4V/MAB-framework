import argparse
import sys
import importlib

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Bandits CLI")

    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run an experiment script")
    run_parser.add_argument("script", help="Python file in multi_agent_bandits/experiments/ (without .py)")
    run_parser.add_argument("--steps", type=int, default=None)
    run_parser.add_argument("--save", type=str, default=None)
    run_parser.add_argument("--plot-rewards", action="store_true")
    run_parser.add_argument("--plot-frequencies", action="store_true")
    run_parser.add_argument("--plot-beliefs", action="store_true")
    run_parser.add_argument("--plot-environment-health", action="store_true")
    run_parser.add_argument("--plot-resource-efficiency", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        module_path = f"multi_agent_bandits.experiments.{args.script}"

        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError:
            print(f"Error: Could not find experiment '{args.script}'.")
            print(f"Make sure there is a file: multi_agent_bandits/experiments/{args.script}.py")
            sys.exit(1)

        if not hasattr(mod, "main"):
            print(f"Error: '{args.script}.py' does not define a main() function.")
            sys.exit(1)

        kwargs = {}
        if args.steps is not None:
            kwargs["steps"] = args.steps
        if args.save is not None:
            kwargs["save_dir"] = args.save
        if args.plot_rewards:
            kwargs["plot_rewards"] = True
        if args.plot_frequencies:
            kwargs["plot_frequencies"] = True
        if args.plot_beliefs:
            kwargs["plot_beliefs"] = True
        if args.plot_environment_health:
            kwargs["plot_environment_health"] = True
        if args.plot_resource_efficiency:
            kwargs["plot_resource_efficiency"] = True
        try:
            mod.main(**kwargs)
        except TypeError:
            mod.main()

        return

    parser.print_help()
