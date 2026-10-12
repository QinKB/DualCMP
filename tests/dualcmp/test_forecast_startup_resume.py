"""The native startup event must also fire after a checkpoint resume."""
import ast
from pathlib import Path


def test_first_optimizer_update_after_resume():
    source = Path(__file__).resolve().parents[2] / "opencood/tools/dualcmp/train_forecast_ddp.py"
    tree = ast.parse(source.read_text())
    condition = next(node.test for node in ast.walk(tree)
        if isinstance(node, ast.If) and any(
            isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "startup"
                for target in statement.targets)
            for statement in node.body))
    expression = compile(ast.Expression(condition), str(source), "eval")
    for initial_step in (0, 12786, 20445):
        for step in (initial_step, initial_step + 1, initial_step + 2):
            for rank in (0, 1):
                assert eval(expression, {}, dict(step=step, initial_step=initial_step, rank=rank)) == (
                    step == initial_step + 1 and rank == 0)
