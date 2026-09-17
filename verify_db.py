import lancedb
import pandas as pd

def inspect_database():
    db = lancedb.connect(".lancedb")
    table = db.open_table("environmental_knowledge")
    df = table.to_pandas()

    print("=== KNOWLEDGE BASE SUMMARY ===")
    print(f"Total Chunks: {len(df)}\n")

    print("--- Chunks per Document ---")
    # Show how many chunks came from each PDF
    doc_counts = df['document_title'].value_counts()
    for doc, count in doc_counts.items():
        print(f"- {count:4d} chunks : {doc}")

    print("\n--- Data Quality Check (1 Random Snippet per Document) ---")
    # Group by document and sample 1 random row from each
    sample_df = df.groupby('document_title').sample(n=1, random_state=42)

    for _, row in sample_df.iterrows():
        print(f"\n📄 {row['document_title']} (Org: {row['source_org']}, Year: {row['year']})")
        print(f"   Page (PDF/Printed): {row['pdf_page']} / {row['printed_page']}")
        print(f"   Domains: {', '.join(row['domains'])}")
        
        # Clean up the text snippet for terminal reading
        snippet = row['text'][:200].replace('\n', ' ')
        print(f"   Snippet: \"{snippet}...\"")
        print("-" * 70)

if __name__ == "__main__":
    pd.set_option('display.max_colwidth', None)
    inspect_database()