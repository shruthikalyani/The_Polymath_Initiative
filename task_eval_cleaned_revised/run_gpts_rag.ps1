$DATA_FILE_PATH = "C:\Users\neela\Downloads\The_Polymath_Initiative\locomo10.json"
$OUT_DIR = "C:\Users\neela\Downloads\The_Polymath_Initiative\results_rag"
$QA_OUTPUT_FILE = "rag_results_summaries_observations_full.json"
# ------------------------------ #

# Your model and provider
$MODEL = "meta-llama/Llama-3.1-8B-Instruct"
$PROVIDER = "nvidia"

# Create output directory if not exists
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

# dialog as database
foreach ($TOP_K in 5, 10, 25, 50) {
    python run_eval.py `
        --data-file $DATA_FILE_PATH --out-file "$OUT_DIR/$QA_OUTPUT_FILE" `
        --model $MODEL --provider $PROVIDER --batch-size 1 --use-rag --retriever dragon --top-k $TOP_K `
        --rag-mode dialog --access-mode full
}

# observation as database
foreach ($TOP_K in 5, 10, 25, 50) {
    python run_eval.py `
        --data-file $DATA_FILE_PATH --out-file "$OUT_DIR/$QA_OUTPUT_FILE" `
        --model $MODEL --provider $PROVIDER --batch-size 1 --use-rag --retriever dragon --top-k $TOP_K `
        --rag-mode observation --access-mode full
}

# summary as database
foreach ($TOP_K in 2, 5, 10) {
    python run_eval.py `
        --data-file $DATA_FILE_PATH --out-file "$OUT_DIR/$QA_OUTPUT_FILE" `
        --model $MODEL --provider $PROVIDER --batch-size 1 --use-rag --retriever dragon --top-k $TOP_K `
        --rag-mode summary --access-mode full
}