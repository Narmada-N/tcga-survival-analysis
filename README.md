
# Survival Analysis on TCGA Data with PyTorch

This script performs survival analysis on TCGA data using a PyTorch implementation. The model architecture is based on the MIL (Multiple Instance Learning) with attention mechanism for survival prediction. The embeddings are extracted based on SSL technique using dinov2
[Histopathology Embeddings Dino-v2](https://github.com/datma-health/dinov2)

## Prerequisites

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- scikit-survival
- tensorboardX

## How to Run

1. Ensure you have all the required libraries installed.
2. Prepare your dataset in the CSV format and set the correct path in the `csv_path` variable.
3. Adjust the configuration parameters as needed using the argparse arguments.
4. Run the script:

```
python main.py --data_root_dir [DATA_DIR] --results_dir [RESULTS_DIR] ... [other args]
```

Replace `main.py` with the name of the script file, and provide appropriate values for `[DATA_DIR]`, `[RESULTS_DIR]`, and any other desired arguments.

## Arguments

- `--data_root_dir`: Data directory to WSI features.
- `--num_classes`: Classes based on survival months.
- `--maxepoch`: Maximum number of epochs.
- `--feats_size`: Feature embedding size from VITL.
- `--non_linearity`: Additional nonlinear operation.
- `--lr`: Learning rate.
- `--results_dir`: Directory to save results.
- `--writer_dir`: Directory for tensorboard logs.
- `--reg`: L2-regularization weight decay.

## Model

The implemented model is `MIL_Attention_FC_surv`, which utilizes the attention mechanism to weigh the importance of different instances in a bag for survival prediction.

## Outputs

The script saves the trained model weights to the specified results directory and logs metrics such as the c-index and losses to tensorboard for visualization.
