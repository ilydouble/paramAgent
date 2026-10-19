# Frozen embedding-3 net-gain router

Train CPU Ridge on cached two-view embeddings to predict guided_success - actor_success in {-1,0,+1}. See experiment SUMMARY.md for verified results and limitations.

python dataset/router/train_embedding_gain_ridge.py train --vectors VECTORS_DIR --run-dir PAIRED_RUN_DIR --output-dir NEW_OUTPUT_DIR

python dataset/router/train_embedding_gain_ridge.py predict --vectors VECTORS_DIR --model OUTPUT_DIR/router.joblib --output NEW_PREDICTIONS.jsonl

Cache contract: vectors.npz (question, answer, sample_ids), metadata.json, manifest.json with SHA256. Each view is 1024-dim L2-normalized embedding-3; UTF-8 chunks <=2800 bytes, byte-weighted mean then L2 normalization. Only whitelisted online features enter the matrix. API key stays in local config; cached inference needs no API key. Alpha and threshold selected only from train OOF; validation is report-only. Output directory must not exist.
