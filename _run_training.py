import sys
sys.path.insert(0, '.')
from src.model_training import run_full_training
result = run_full_training()
print('Best model:', result['best_model'])
print()
for m in result['all_metrics']:
    print(f"{m['model']:25s} Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
    print(f"  Confusion matrix: {m['confusion_matrix']}")
