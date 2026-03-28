"""Gradio dashboard for the Cosmic Signal Processing Pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import gradio as gr

from dashboard.handlers import export_cleaned, generate_data, run_pipeline_ui, start_game_pipeline, upload_csv

logging.basicConfig(level=logging.INFO)

# ── Theme ─────────────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_700",
    block_label_text_color="*neutral_200",
    block_title_text_color="*neutral_100",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
)

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Cosmic Pipeline") as app:

    gr.Markdown("# 🛰️ Kozmik Veri Ayiklama ve Isleme Hatti")
    gr.Markdown("**TUA Astro Hackathon 2026** — Ahmet & Omer")

    with gr.Tabs():

        # ── TAB 1: Veri ──────────────────────────────
        with gr.Tab("📡 Veri"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Sentetik Veri Parametreleri")
                    sl_n = gr.Slider(1000, 20000, value=5000, step=1000, label="Veri Noktasi")
                    num_seed = gr.Number(value=42, label="Seed", precision=0)
                    sl_seu = gr.Slider(5, 50, value=15, step=1, label="SEU (Bit-flip) Sayisi")
                    sl_tid = gr.Slider(0.001, 0.01, value=0.003, step=0.001, label="TID Drift Egimi")
                    sl_gap = gr.Slider(1, 10, value=4, step=1, label="Veri Boslugu Sayisi")
                    sl_noise = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Gurultu Seviyesi")
                    btn_gen = gr.Button("🔬 Sentetik Veri Uret", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("### CSV Yukle")
                    csv_file = gr.File(
                        label="CSV/TSV/Excel/JSON/HDF5/Parquet",
                        file_types=[".csv", ".tsv", ".xlsx", ".xls", ".json", ".h5", ".hdf5", ".parquet"],
                    )
                    btn_csv = gr.Button("📁 CSV Yukle")
                    chk_columns = gr.CheckboxGroup(
                        choices=[], value=[], visible=False,
                        label="Kanallar (Multi-Channel)",
                    )

                    txt_status = gr.Textbox(label="Durum", interactive=False)

                with gr.Column(scale=3):
                    plot_preview = gr.Plot(label="Sinyal Onizleme")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Temiz Veri")
                            tbl_clean = gr.Dataframe(label="Ground Truth", max_height=300)
                        with gr.Column():
                            gr.Markdown("#### Bozuk Veri")
                            tbl_corrupt = gr.Dataframe(label="Radyasyonlu", max_height=300)

            btn_gen.click(
                fn=generate_data,
                inputs=[sl_n, num_seed, sl_seu, sl_tid, sl_gap, sl_noise],
                outputs=[plot_preview, tbl_clean, tbl_corrupt, txt_status],
            )
            btn_csv.click(
                fn=upload_csv,
                inputs=[csv_file],
                outputs=[plot_preview, tbl_clean, tbl_corrupt, txt_status, chk_columns],
            )

        # ── TAB 2: Pipeline ──────────────────────────
        with gr.Tab("🔧 Pipeline"):
            with gr.Row():
                radio_method = gr.Radio(
                    choices=["classic", "ml", "both"],
                    value="classic",
                    label="Pipeline Yontemi",
                )
                btn_run = gr.Button(
                    "▶️ Pipeline Calistir", variant="primary", size="lg",
                )

            txt_metrics = gr.Textbox(label="Metrikler", interactive=False)

            with gr.Tabs():
                with gr.Tab("📈 Sonuc Grafigi"):
                    plot_result = gr.Plot(label="Ground Truth vs Bozuk vs Temizlenmis")

                with gr.Tab("🔍 Detektor Detayi"):
                    plot_det = gr.Plot(label="Her Dedektorun Buldugu Anomaliler")

                with gr.Tab("📋 Islem Raporu"):
                    code_log = gr.Code(label="Pipeline Log", language=None, lines=30)

                with gr.Tab("📊 Anomali Tablosu"):
                    tbl_faults = gr.Dataframe(label="Tespit Edilen Anomaliler", max_height=400)

                with gr.Tab("🔍 Dogrulama"):
                    txt_verify = gr.Textbox(
                        label="Onarim Dogrulama", interactive=False, lines=3,
                    )

                with gr.Tab("🔬 Adim Adim Izleme"):
                    tbl_tracer = gr.Dataframe(label="Pipeline Adim Tablosu", max_height=500)
                    code_tracer = gr.Code(label="Ozet Rapor", language=None, lines=25)

            btn_run.click(
                fn=run_pipeline_ui,
                inputs=[radio_method, chk_columns],
                outputs=[plot_result, code_log, plot_det, txt_metrics, tbl_faults, txt_verify, tbl_tracer, code_tracer],
            )

            with gr.Row():
                btn_export = gr.Button("💾 Temizlenmis Veriyi Indir", variant="secondary")
                file_export = gr.File(label="Indirme", visible=False)
            btn_export.click(fn=export_cleaned, inputs=[], outputs=[file_export])

        # ── TAB 3: DataCraft ─────────────────────────
        with gr.Tab("🎮 DataCraft"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎮 Oyun Kontrolleri")
                    game_samples = gr.Slider(
                        minimum=500, maximum=10000, value=2000, step=500,
                        label="Veri Noktasi",
                    )
                    game_method = gr.Radio(
                        choices=["classic", "both"],
                        value="classic",
                        label="Pipeline Yontemi",
                    )
                    btn_game = gr.Button(
                        "🚀 Oyuna Basla", variant="primary", size="lg",
                    )
                    game_status = gr.Textbox(
                        label="Pipeline Durumu", interactive=False,
                    )
                    game_result = gr.HTML(label="Sonuc")

            _game_abs = str(Path(__file__).parent / "game.html")
            gr.HTML(
                value=f'<iframe src="/file={_game_abs}" width="100%" height="600" style="border:none; border-radius:12px;"></iframe>',
            )

            btn_game.click(
                fn=start_game_pipeline,
                inputs=[game_samples, game_method],
                outputs=[game_result, game_status],
            )


if __name__ == "__main__":
    app.launch(
        server_port=7860,
        share=False,
        theme=theme,
        allowed_paths=[str(Path(__file__).parent / "game.html")],
    )
