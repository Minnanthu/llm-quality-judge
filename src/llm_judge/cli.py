"""CLI commands for the LLM evaluation pipeline."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="llm-judge",
    help="LLM evaluation pipeline: inference → autocheck → judge → compare",
)
console = Console()

DEFAULT_CONFIG = "configs/run-config.yaml"


@app.command()
def infer(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSONL path"),
) -> None:
    """Stage 1: Run candidate model inference."""
    from llm_judge.stages.inference import run_inference

    console.print("[bold blue]Stage 1: Inference[/bold blue]")
    out = run_inference(config, output)
    console.print(f"[green]Done:[/green] {out}")


@app.command()
def autocheck(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
    inference: Optional[str] = typer.Option(None, "--inference", "-i", help="Inference JSONL path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSONL path"),
) -> None:
    """Stage 2: Run automated format and schema checks."""
    from llm_judge.stages.autocheck import run_autocheck

    console.print("[bold blue]Stage 2: Autocheck[/bold blue]")
    out = run_autocheck(config, inference, output)
    console.print(f"[green]Done:[/green] {out}")


@app.command()
def judge(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
    inference: Optional[str] = typer.Option(None, "--inference", "-i", help="Inference JSONL path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSONL path"),
) -> None:
    """Stage 3: Run LLM-as-a-Judge evaluation."""
    from llm_judge.stages.judge import run_judge

    console.print("[bold blue]Stage 3: Judge[/bold blue]")
    out = run_judge(config, inference, output)
    console.print(f"[green]Done:[/green] {out}")


@app.command()
def compare(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
    judgements: Optional[str] = typer.Option(None, "--judgements", "-j", help="Judgements JSONL path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON path"),
) -> None:
    """Stage 4: Aggregate and produce comparison report."""
    from llm_judge.stages.compare import run_compare

    console.print("[bold blue]Stage 4: Compare[/bold blue]")
    out = run_compare(config, judgements, output_path=output)
    console.print(f"[green]Done:[/green] {out}")
    console.print(f"[green]Markdown:[/green] {out.with_suffix('.md')}")


@app.command()
def run_all(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
) -> None:
    """Run all 4 stages in sequence."""
    from llm_judge.stages.autocheck import run_autocheck
    from llm_judge.stages.compare import run_compare
    from llm_judge.stages.inference import run_inference
    from llm_judge.stages.judge import run_judge

    console.print("[bold magenta]Running full pipeline[/bold magenta]")

    console.print("\n[bold blue]Stage 1: Inference[/bold blue]")
    inf_out = run_inference(config)
    console.print(f"[green]Done:[/green] {inf_out}")

    console.print("\n[bold blue]Stage 2: Autocheck[/bold blue]")
    ac_out = run_autocheck(config, str(inf_out))
    console.print(f"[green]Done:[/green] {ac_out}")

    console.print("\n[bold blue]Stage 3: Judge[/bold blue]")
    jdg_out = run_judge(config, str(inf_out))
    console.print(f"[green]Done:[/green] {jdg_out}")

    console.print("\n[bold blue]Stage 4: Compare[/bold blue]")
    cmp_out = run_compare(config, str(jdg_out))
    console.print(f"[green]Done:[/green] {cmp_out}")
    console.print(f"[green]Markdown:[/green] {cmp_out.with_suffix('.md')}")

    console.print("\n[bold green]Pipeline complete![/bold green]")
