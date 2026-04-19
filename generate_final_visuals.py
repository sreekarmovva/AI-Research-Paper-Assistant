import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visuals():
    df = pd.read_csv("final_single_doc_results.csv")
    df['Question_Index'] = (df.index // 2) + 1 # Correctly assign indices for interleaved Baseline/RAG rows
    
    # Professional styling
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    
    # 1. Overall Summary (Averages)
    summary = df.groupby('System').mean(numeric_only=True).reset_index()
    summary.to_csv("final_single_doc_summary.csv", index=False)

    # ---------------------------------------------------------
    # 2. BERTScore: Bar + Trend
    # ---------------------------------------------------------
    # Bar Chart (No Error Bars)
    plt.figure(figsize=(7, 6))
    ax1 = sns.barplot(data=df, x='System', y='BERTScore', palette="Blues_d", errorbar=None, width=0.6)
    plt.title("Semantic Similarity Performance (BERTScore)")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.05)
    for c in ax1.containers: ax1.bar_label(c, fmt='%.3f', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig("bertscore_comparison_bar.png", dpi=300)
    plt.close()

    # Trend Graph
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x=df.index // 2, y='BERTScore', hue='System', marker='o', palette="Blues_d")
    plt.title("BERTScore Trend Across Evaluation Questions")
    plt.xlabel("Question Index")
    plt.ylabel("BERTScore F1")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("bertscore_trend.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. ROUGE & BERTScore Comparison: Bar + Trend
    # ---------------------------------------------------------
    # Comparison Bar (BERTScore vs ROUGE_L)
    comp_df = df.melt(id_vars=['System'], value_vars=['BERTScore', 'ROUGE_L'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(data=comp_df, x='Metric', y='Score', hue='System', palette="GnBu_d", errorbar=None)
    plt.title("Comparison: BERTScore (Semantic) vs ROUGE-L (Lexical)")
    plt.ylim(0, 1.05)
    for c in ax2.containers: ax2.bar_label(c, fmt='%.3f', padding=5)
    plt.tight_layout()
    plt.savefig("rouge_vs_bertscore_bar.png", dpi=300)
    plt.close()

    # Correlation / Dual Trend
    rag_only = df[df['System'] == 'RAG System'].reset_index()
    fig, ax_b = plt.subplots(figsize=(12, 6))
    ax_r = ax_b.twinx()
    
    sns.lineplot(data=rag_only, x=rag_only.index, y='BERTScore', ax=ax_b, color='blue', marker='o', label='BERTScore (Left)')
    sns.lineplot(data=rag_only, x=rag_only.index, y='ROUGE_L', ax=ax_r, color='green', marker='s', label='ROUGE-L (Right)')
    
    ax_b.set_ylabel("BERTScore F1", color='blue')
    ax_r.set_ylabel("ROUGE-L F-measure", color='green')
    ax_b.set_ylim(0.7, 1.0)
    ax_r.set_ylim(0, 0.5)
    plt.title("RAG Performance Dynamics: BERTScore vs ROUGE-L")
    ax_b.set_xlabel("Question Index")
    plt.tight_layout()
    plt.savefig("rouge_bertscore_dual_trend.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. LLM-as-a-Judge: Bar + Trend
    # ---------------------------------------------------------
    metrics_judge = ['Relevance', 'Accuracy', 'Completeness', 'Groundedness']
    df_melt_judge = df.melt(id_vars=['System'], value_vars=metrics_judge, var_name='Metric', value_name='Score')
    
    # Aggregate Bar
    plt.figure(figsize=(12, 7))
    ax3 = sns.barplot(data=df_melt_judge, x='Metric', y='Score', hue='System', palette="vlag", errorbar=None)
    plt.title("LLM-as-a-Judge Evaluation Summary")
    plt.ylabel("Score (1 to 5)")
    plt.ylim(0, 5.5)
    for c in ax3.containers: ax3.bar_label(c, fmt='%.2f', padding=5)
    plt.legend(title='Architecture', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("judge_metrics_comparison_bar.png", dpi=300)
    plt.close()

    # Multi-metric Trend for RAG
    plt.figure(figsize=(14, 6))
    rag_melt = rag_only.melt(id_vars=['Question_Index'], value_vars=metrics_judge, var_name='Metric', value_name='Score')
    sns.lineplot(data=rag_melt, x='Question_Index', y='Score', hue='Metric', marker='o', palette="Set2")
    plt.title("RAG Consistency: LLM Judge Metrics Across Questions")
    plt.ylabel("Score")
    plt.ylim(0, 5.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("judge_metrics_trend.png", dpi=300)
    plt.close()
    
    print("\n[OK] Professional Research Graphics Generated:")
    print(" -> bertscore_comparison_bar.png / bertscore_trend.png")
    print(" -> rouge_vs_bertscore_bar.png / rouge_bertscore_dual_trend.png")
    print(" -> judge_metrics_comparison_bar.png / judge_metrics_trend.png")

if __name__ == "__main__":
    generate_visuals()
