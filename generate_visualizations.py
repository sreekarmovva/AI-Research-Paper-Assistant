import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations():
    print("Loading data from evaluation_results.csv...")
    try:
        df = pd.read_csv('evaluation_results.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Calculate arithmetic means across all numeric fields grouped by System
    summary_df = df.groupby('System').mean(numeric_only=True).reset_index()

    # Set up global plot styles
    sns.set_theme(style="whitegrid")

    # =======================================================
    # 1. BERTScore Comparison (Bar Chart)
    # =======================================================
    print("Generating BERTScore plot...")
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(data=summary_df, x='System', y='BERTScore_F1', palette="Blues_d")
    plt.title('Average BERTScore (F1) Comparison', pad=15)
    plt.ylabel('BERTScore F1 (0 - 1.0)')
    plt.ylim(0, 1.0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.savefig('chart_bertscore_comparison.png', dpi=300)
    plt.close()

    # =======================================================
    # 2. LLM Evaluation Metrics (Grouped Bar Chart)
    # =======================================================
    print("Generating LLM evaluation metrics plot...")
    metrics = ['Relevance', 'Accuracy', 'Completeness', 'Groundedness']
    # Melt the dataframe strictly for plotting grouped bars easily
    df_melt = df.melt(id_vars=['System'], value_vars=metrics, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(data=df_melt, x='Metric', y='Score', hue='System', palette="vlag", errorbar=None)
    plt.title('LLM-as-a-Judge Evaluation Metrics (Average)', pad=15)
    plt.ylabel('Score (1 - 5)')
    plt.ylim(0, 5)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2f', padding=3)
    plt.legend(title='System', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('chart_llm_metrics_comparison.png', dpi=300)
    plt.close()

    # =======================================================
    # 3. Latency Comparison (Bar Chart)
    # =======================================================
    print("Generating Latency plot...")
    plt.figure(figsize=(6, 5))
    ax3 = sns.barplot(data=summary_df, x='System', y='Latency_sec', palette="Reds_d")
    plt.title('Average Latency Comparison', pad=15)
    plt.ylabel('Latency (Seconds)')
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.2fs', padding=3)
    plt.tight_layout()
    plt.savefig('chart_latency_comparison.png', dpi=300)
    plt.close()

    # =======================================================
    # 4. Summary Table Generation (CSV)
    # =======================================================
    print("Generating Summary Table CSV...")
    # Reshaping so columns are Baseline / RAG and Rows are Metrics
    summary_transposed = summary_df.set_index('System').T
    summary_transposed.index.name = 'Metric'
    summary_transposed.reset_index(inplace=True)
    
    table_filename = 'summary_table.csv'
    summary_transposed.to_csv(table_filename, index=False)
    
    print("\n✅ All visualization tasks complete!")
    print(f"-> Saved: chart_bertscore_comparison.png")
    print(f"-> Saved: chart_llm_metrics_comparison.png")
    print(f"-> Saved: chart_latency_comparison.png")
    print(f"-> Saved: {table_filename}")

if __name__ == "__main__":
    generate_visualizations()
