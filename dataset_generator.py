#!/usr/bin/env python3
"""
Textual TUI App for Moltbook Captcha Dataset Generator

Features:
- Harvest captchas from random posts
- Generate multiple sample captchas
- View dataset statistics
- Split train/eval sets
- Template-based response builder
"""

import os
import asyncio
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label, 
    DataTable, Select, RadioSet, RadioButton, ProgressBar,
    TextArea, TabbedContent, TabPane, Checkbox, ListView, ListItem
)
from textual.binding import Binding
from textual.screen import Screen
from textual.reactive import reactive

from captcha_harvester import (
    DatasetGenerator,
    generate_multiple_captchas,
    fetch_random_posts,
    harvest_captchas_from_posts,
    load_dataset,
    get_responded_post_ids,
    DATASET_FILE
)


class StatsScreen(Screen):
    """Screen for viewing dataset statistics."""
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Label("Dataset Statistics", classes="title"),
            Static(id="stats_content"),
            Button("Back", id="back_btn", variant="primary"),
            classes="container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        self.update_stats()
    
    def update_stats(self) -> None:
        generator = DatasetGenerator()
        stats = generator.get_stats()
        
        content = f"""
Total Entries: {stats['total_entries']}
With Answers: {stats['with_answers']}
With Errors: {stats['with_errors']}
Success Rate: {stats['success_rate']:.1f}%

Dataset Location: {DATASET_FILE}
        """
        self.query_one("#stats_content", Static).update(content)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()


class HarvestScreen(Screen):
    """Screen for harvesting captchas from posts."""
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Label("Harvest Captchas from Posts", classes="title"),
            
            Label("Moltbook API Key:"),
            Input(placeholder="Enter your Moltbook API key", id="api_key", password=True),
            
            Label("OpenAI API Key:"),
            Input(placeholder="Enter your OpenAI API key", id="openai_key", password=True),
            
            Label("Number of Posts:"),
            Input(value="10", id="num_posts"),
            
            Checkbox("Lean into niche posts (lower engagement)", id="niche_checkbox", value=True),
            
            Label("AI Model:"),
            Select([
                ("gpt-4.1-nano", "gpt-4.1-nano"),
                ("gpt-4.1-mini", "gpt-4.1-mini"),
                ("gpt-4o-mini", "gpt-4o-mini"),
            ], id="model_select", value="gpt-4.1-nano"),
            
            ProgressBar(id="harvest_progress", show_eta=False, show_percentage=True),
            
            Static(id="harvest_status"),
            
            Horizontal(
                Button("Start Harvesting", id="start_btn", variant="success"),
                Button("Back", id="back_btn", variant="primary"),
            ),
            classes="container"
        )
        yield Footer()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "start_btn":
            await self.start_harvesting()
    
    async def start_harvesting(self) -> None:
        api_key = self.query_one("#api_key", Input).value
        openai_key = self.query_one("#openai_key", Input).value
        num_posts = int(self.query_one("#num_posts", Input).value or 10)
        lean_niche = self.query_one("#niche_checkbox", Checkbox).value
        model = self.query_one("#model_select", Select).value
        
        if not api_key or not openai_key:
            self.query_one("#harvest_status", Static).update(
                "[red]Error: Both API keys are required[/red]"
            )
            return
        
        progress = self.query_one("#harvest_progress", ProgressBar)
        status = self.query_one("#harvest_status", Static)
        
        progress.update(total=num_posts)
        status.update("Fetching random posts...")
        
        # Fetch posts
        posts = fetch_random_posts(api_key, num_posts, lean_niche)
        
        if not posts:
            status.update("[red]No unresponded posts found![/red]")
            return
        
        status.update(f"Found {len(posts)} posts. Solving captchas...")
        
        # Harvest captchas
        for i, post in enumerate(posts):
            try:
                results = harvest_captchas_from_posts(
                    api_key, openai_key, [post], 
                    model=model if isinstance(model, str) else "gpt-4.1-nano"
                )
                progress.advance(1)
                status.update(f"Processed {i+1}/{len(posts)} posts...")
                await asyncio.sleep(0.1)  # Allow UI to update
            except Exception as e:
                status.update(f"[red]Error on post {i+1}: {e}[/red]")
        
        status.update(f"[green]✓ Harvested {len(posts)} captchas![/green]")


class GenerateScreen(Screen):
    """Screen for generating multiple sample captchas."""
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Label("Generate Sample Captchas", classes="title"),
            
            Label("OpenAI API Key:"),
            Input(placeholder="Enter your OpenAI API key", id="openai_key", password=True),
            
            Label("Number of Captchas:"),
            Input(value="10", id="num_captchas"),
            
            Label("AI Model:"),
            Select([
                ("gpt-4.1-nano", "gpt-4.1-nano"),
                ("gpt-4.1-mini", "gpt-4.1-mini"),
                ("gpt-4o-mini", "gpt-4o-mini"),
            ], id="model_select", value="gpt-4.1-nano"),
            
            ProgressBar(id="gen_progress", show_eta=False, show_percentage=True),
            
            Static(id="gen_status"),
            
            Horizontal(
                Button("Start Generation", id="start_btn", variant="success"),
                Button("Back", id="back_btn", variant="primary"),
            ),
            classes="container"
        )
        yield Footer()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "start_btn":
            await self.start_generation()
    
    async def start_generation(self) -> None:
        openai_key = self.query_one("#openai_key", Input).value
        num_captchas = int(self.query_one("#num_captchas", Input).value or 10)
        model = self.query_one("#model_select", Select).value
        
        if not openai_key:
            self.query_one("#gen_status", Static).update(
                "[red]Error: OpenAI API key is required[/red]"
            )
            return
        
        progress = self.query_one("#gen_progress", ProgressBar)
        status = self.query_one("#gen_status", Static)
        
        progress.update(total=num_captchas)
        status.update("Generating captchas...")
        
        # Generate captchas
        for i in range(num_captchas):
            try:
                results = generate_multiple_captchas(
                    openai_key, 
                    count=1,
                    model=model if isinstance(model, str) else "gpt-4.1-nano"
                )
                progress.advance(1)
                status.update(f"Generated {i+1}/{num_captchas} captchas...")
                await asyncio.sleep(0.1)
            except Exception as e:
                status.update(f"[red]Error on captcha {i+1}: {e}[/red]")
        
        status.update(f"[green]✓ Generated {num_captchas} captchas![/green]")


class SplitScreen(Screen):
    """Screen for splitting dataset into train/eval sets."""
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Label("Train/Eval Split", classes="title"),
            
            Label("Output Directory:"),
            Input(value="./dataset_output", id="output_dir"),
            
            Label("Train Ratio (0.0 - 1.0):"),
            Input(value="0.8", id="train_ratio"),
            
            Label("Random Seed:"),
            Input(value="42", id="seed"),
            
            Label("Response Template (optional):"),
            Input(
                placeholder="e.g., {compressed} {equation} = {answer}",
                id="response_template"
            ),
            
            Static(id="split_status"),
            
            Horizontal(
                Button("Split Dataset", id="split_btn", variant="success"),
                Button("Back", id="back_btn", variant="primary"),
            ),
            classes="container"
        )
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "split_btn":
            self.split_dataset()
    
    def split_dataset(self) -> None:
        output_dir = self.query_one("#output_dir", Input).value
        train_ratio = float(self.query_one("#train_ratio", Input).value or 0.8)
        seed = int(self.query_one("#seed", Input).value or 42)
        template = self.query_one("#response_template", Input).value or None
        
        status = self.query_one("#split_status", Static)
        
        try:
            generator = DatasetGenerator()
            
            if not generator.entries:
                status.update("[red]Error: No entries in dataset![/red]")
                return
            
            result = generator.export_train_eval(
                output_dir,
                train_ratio=train_ratio,
                response_template=template,
                seed=seed
            )
            
            status.update(f"""[green]✓ Dataset split complete!
            
Train set: {result['train_count']} entries -> {result['train_path']}
Eval set: {result['eval_count']} entries -> {result['eval_path']}[/green]""")
        
        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")


class TemplateBuilderScreen(Screen):
    """Screen for building custom response templates."""
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b", "back", "Back"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Label("Response Template Builder", classes="title"),
            
            Label("Available Variables:"),
            Static("""
{compressed} - Compressed reasoning from AI
{equation} - Extracted equation
{answer} - Final numeric answer
{preprocessed} - Cleaned captcha text
{uncleaned} - Raw captcha text
{full_response} - Complete AI response
            """),
            
            Label("Template:"),
            Input(
                value="{compressed}\n{equation}\nAnswer: {answer}",
                id="template_input"
            ),
            
            Label("Preview (first entry):"),
            Static(id="template_preview", classes="preview"),
            
            Horizontal(
                Button("Update Preview", id="preview_btn", variant="primary"),
                Button("Back", id="back_btn", variant="primary"),
            ),
            classes="container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        self.update_preview()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "preview_btn":
            self.update_preview()
    
    def update_preview(self) -> None:
        template = self.query_one("#template_input", Input).value
        preview = self.query_one("#template_preview", Static)
        
        generator = DatasetGenerator()
        
        if not generator.entries:
            preview.update("[red]No entries in dataset to preview![/red]")
            return
        
        # Get first entry with an answer
        entry = None
        for e in generator.entries:
            if e.get('answer'):
                entry = e
                break
        
        if not entry:
            entry = generator.entries[0]
        
        try:
            formatted = generator.build_response(entry, template)
            preview.update(f"""Template: {template}

--- Preview ---
{formatted}

--- Original Data ---
Uncleaned: {entry.get('uncleaned', 'N/A')[:100]}...
Answer: {entry.get('answer', 'N/A')}""")
        except Exception as e:
            preview.update(f"[red]Error: {e}[/red]")


class DatasetGeneratorApp(App):
    """Main Textual TUI Application."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    
    .container {
        width: 80%;
        height: auto;
        padding: 1 2;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .preview {
        border: solid green;
        padding: 1;
        margin: 1 0;
    }
    
    Input {
        margin-bottom: 1;
    }
    
    Select {
        margin-bottom: 1;
    }
    
    Button {
        margin-right: 1;
    }
    
    ProgressBar {
        margin: 1 0;
    }
    
    Static {
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Label("Moltbook Captcha Dataset Generator", classes="title"),
            
            Static("""
Welcome to the Moltbook Captcha Dataset Generator!

This tool helps you:
- Harvest captchas from real Moltbook posts
- Generate sample captchas for testing
- Split datasets into train/eval sets
- Build custom response templates

Choose an option below:
            """),
            
            Grid(
                Button("📊 View Statistics", id="stats_btn", variant="primary"),
                Button("🌾 Harvest from Posts", id="harvest_btn", variant="success"),
                Button("🎲 Generate Samples", id="generate_btn", variant="success"),
                Button("✂️ Train/Eval Split", id="split_btn", variant="warning"),
                Button("🔧 Template Builder", id="template_btn", variant="info"),
                classes="grid"
            ),
            
            Static(id="dataset_info"),
            classes="container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        self.update_dataset_info()
    
    def update_dataset_info(self) -> None:
        generator = DatasetGenerator()
        stats = generator.get_stats()
        
        info = self.query_one("#dataset_info", Static)
        info.update(f"""
Current Dataset: {stats['total_entries']} entries
Success Rate: {stats['success_rate']:.1f}%
Location: {DATASET_FILE}
        """)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id == "stats_btn":
            self.push_screen(StatsScreen())
        elif button_id == "harvest_btn":
            self.push_screen(HarvestScreen())
        elif button_id == "generate_btn":
            self.push_screen(GenerateScreen())
        elif button_id == "split_btn":
            self.push_screen(SplitScreen())
        elif button_id == "template_btn":
            self.push_screen(TemplateBuilderScreen())


if __name__ == "__main__":
    app = DatasetGeneratorApp()
    app.run()
