"""Command-line interface for the Cosmic Pipeline."""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Cosmic Pipeline CLI")
    sub = parser.add_subparsers(dest="command")

    proc = sub.add_parser("process", help="CSV dosyasini isle")
    proc.add_argument("input", help="Giris CSV dosyasi")
    proc.add_argument("--output", "-o", default="cleaned_output.csv")
    proc.add_argument("--method", "-m", choices=["classic", "ml", "both"], default="classic")
    proc.add_argument("--multi", action="store_true", help="Tum numeric sutunlari isle")

    info = sub.add_parser("info", help="CSV hakkinda bilgi")
    info.add_argument("input", help="CSV dosyasi")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "info":
        df = pd.read_csv(args.input)
        print(f"Satir: {len(df)}, Sutun: {len(df.columns)}")
        print(f"Sutunlar: {list(df.columns)}")
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        print(f"Numeric: {numeric}")
        print(f"NaN: {df.isna().sum().sum()}")
        for col in numeric[:5]:
            print(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}")

    elif args.command == "process":
        from pipeline.orchestrator import run_pipeline, run_pipeline_multi

        df = pd.read_csv(args.input)
        print(f"Yuklendi: {len(df)} satir")

        if args.multi:
            result = run_pipeline_multi(df, method=args.method)
            s = result["summary"]
            print(f"Multi-channel: {s['total_channels']} kanal, {s['total_faults']} anomali, {s['processing_time']:.2f}s")
            for col, faults in s["per_channel"].items():
                print(f"  {col}: {faults} anomali")
        else:
            # Resolve timestamp
            if "timestamp" not in df.columns:
                for a in ("time_tag", "date", "datetime", "time", "ds"):
                    if a in df.columns:
                        df = df.rename(columns={a: "timestamp"})
                        break
                else:
                    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="1s")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Resolve value
            if "value" not in df.columns:
                numeric = [c for c in df.select_dtypes(include=["number"]).columns if c != "timestamp"]
                if numeric:
                    df = df.rename(columns={numeric[0]: "value"})

            result = run_pipeline(df[["timestamp", "value"]], method=args.method)
            m = result["metrics"]
            print(f"Anomali: {m['faults_detected']} | Sure: {m['processing_time']:.2f}s")
            result["cleaned_data"].to_csv(args.output, index=False)
            print(f"Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
