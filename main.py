# main.py
# The Spark of Life (Entry point)

import argparse
import time
import yaml
import torch
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console

from cortex.listener import Listener
from cortex.analyzer import Analyzer
from cortex.vibe_check import VibeCheck
from psyche.personality import Personality
from psyche.memory import ShortTermMemory, flatten_features
from psyche.decision import DecisionMaker
from voice.hallucinator import Hallucinator
from voice.streamer import Streamer

# The number of features extracted by the analyzer.
INPUT_FEATURE_SIZE = 19

def load_config(persona_name):
    """Loads persona and biometric configurations."""
    with open('config/persona.yaml', 'r') as f:
        personas = yaml.safe_load(f)
        persona_config = personas.get(persona_name)
        if not persona_config:
            raise ValueError(f"Persona '{persona_name}' not found in persona.yaml")

    with open('config/bio_metrics.yaml', 'r') as f:
        bio_metrics = yaml.safe_load(f)

    return persona_config, bio_metrics

def create_layout() -> Layout:
    """Defines the layout for the rich display."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main"),
    )
    layout["main"].split_row(Layout(name="state"), Layout(name="log", ratio=2))
    return layout

def main(args):
    """The main loop of Pneuma."""
    console = Console()
    console.print("[bold cyan]--- PNEUMA AWAKENING ---[/bold cyan]")

    # --- Load Configuration ---
    try:
        persona_config, bio_metrics = load_config(args.persona)
        console.print(f"Loaded Persona: [bold yellow]{args.persona}[/bold yellow]")
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[bold red]Error loading configuration: {e}[/bold red]")
        return

    # --- Initialize Components ---
    listener = Listener(sample_rate=bio_metrics['sample_rate'], chunk_size=bio_metrics['chunk_size'])
    analyzer = Analyzer(sample_rate=bio_metrics['sample_rate'])
    vibe_check = VibeCheck()
    personality = Personality(persona_config)
    memory = ShortTermMemory(input_size=INPUT_FEATURE_SIZE, hidden_size=bio_metrics['memory_hidden_size'])
    decision_maker = DecisionMaker()
    hallucinator = Hallucinator(model_path=args.model)
    streamer = Streamer(sample_rate=bio_metrics['sample_rate'])

    layout = create_layout()
    layout["header"].update(Panel("[bold green]PNEUMA is Listening...[/bold green]", subtitle="Press Ctrl+C to exit"))
    
    try:
        with Live(layout, console=console, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            while True:
                # 1. Listen & Analyze
                audio_chunk = listener.listen()
                features = analyzer.analyze(audio_chunk)
                vibe = vibe_check.get_vibe(features)

                # 2. Psychologize
                personality.update(vibe)
                feature_tensor = flatten_features(features)
                _, memory_context = memory.remember(feature_tensor)

                # 3. Decide & Act
                personality_state = personality.get_state()
                intent = decision_maker.decide(personality_state, memory_context)

                # Update UI
                state_panel = Panel(
                    f"[bold]Vibe:[/bold] {vibe}\n"
                    f"[bold]Mood:[/bold] {personality_state['mood']}\n"
                    f"[bold]Co-op:[/bold] {personality_state['cooperativeness']:.2f}\n"
                    f"[bold]Intent:[/bold] [yellow]{intent}[/yellow]",
                    title="[cyan]Psyche State[/cyan]",
                    border_style="cyan"
                )
                layout["state"].update(state_panel)

                # 4. Generate audio
                if intent != "silence":
                    generated_audio = hallucinator.generate(intent, audio_chunk)
                    streamer.play(generated_audio.detach().numpy())
                    layout["log"].update(Panel(f"Generated [bold magenta]{intent}[/bold magenta] audio.", border_style="magenta"))
                else:
                    layout["log"].update(Panel("Silence...", border_style="grey50"))


                time.sleep(bio_metrics.get('reaction_time', 0.1))

    except KeyboardInterrupt:
        console.print("\n[bold cyan]--- PNEUMA IS RETURNING TO SLUMBER ---[/bold cyan]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        listener.stop()
        streamer.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PNEUMA: A Synthetic Bandmate")
    parser.add_argument('--persona', type=str, default="neutral",
                        help='The personality to load (e.g., "heckler", "shadow").')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained voice model checkpoint (.pth).')
    args = parser.parse_args()
    main(args)
