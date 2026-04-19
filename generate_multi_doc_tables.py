import pandas as pd
import os

def generate_multi_doc_tables():
    print("Loading data from multi_doc_evaluation_results.csv...")
    try:
        df = pd.read_csv('multi_doc_evaluation_results.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    def df_to_markdown(df):
        cols = df.columns.tolist()
        header = "| " + " | ".join(map(str, cols)) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(map(lambda x: str(round(x, 4)) if isinstance(x, float) else str(x), row)) + " |")
        return "\n".join([header, separator] + rows)

    # 1. Overall Summary Table (Means for all metrics)
    print("Generating overall summary table...")
    metrics = ['BERTScore_F1', 'Relevance', 'Accuracy', 'Completeness', 'Groundedness', 'Latency_sec']
    overall_summary = df.groupby('System')[metrics].mean().reset_index()
    
    # Transpose for a better paper-style format (Metrics as rows)
    overall_summary_t = overall_summary.set_index('System').T.reset_index()
    overall_summary_t.columns = ['Metric', 'Baseline', 'Multi-Doc RAG']
    
    # Save Overall Summary
    overall_summary_t.to_csv('multi_doc_overall_summary.csv', index=False)
    
    # Markdown version of Overall Summary
    with open('multi_doc_tables.md', 'w') as f:
        f.write("# Multi-Doc Evaluation Summary Tables\n\n")
        f.write("## Overall Comparison (Mean Scores)\n\n")
        f.write(df_to_markdown(overall_summary_t))
        f.write("\n\n")

    # 2. Summary Table for EACH Metric (Mean, Std, Min, Max)
    print("Generating statistical summary for each metric...")
    with open('multi_doc_tables.md', 'a') as f:
        for metric in metrics:
            f.write(f"## {metric} Summary Statistics\n\n")
            metric_stats = df.groupby('System')[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
            # Rounding for cleanliness (done in df_to_markdown as well)
            f.write(df_to_markdown(metric_stats))
            f.write("\n\n")
            
            # Save individual CSVs for convenience
            safe_name = metric.lower().replace(' ', '_')
            metric_stats.round(4).to_csv(f'multi_doc_stats_{safe_name}.csv', index=False)

    print("\n[OK] Multi-Doc tables generated successfully!")
    print("-> Saved: multi_doc_overall_summary.csv")
    print("-> Saved: multi_doc_tables.md (Markdown format for easy copy-pasting)")
    for metric in metrics:
        safe_name = metric.lower().replace(' ', '_')
        print(f"-> Saved: multi_doc_stats_{safe_name}.csv")

if __name__ == "__main__":
    generate_multi_doc_tables()
