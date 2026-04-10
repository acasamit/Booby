# OncoScan

A lightweight, from-scratch Multi-Layer Perceptron (MLP) implemented in pure Lua. Designed to classify breast cancer tumors (Malignant/Benign) using a Wisconsin Diagnostic dataset.

No heavy frameworks, just math and tables.

## Core Features
* **Custom MLP Engine**: Fully configurable hidden layers and neuron counts.
* **Adam Optimizer**: Built-in Adaptive Moment Estimation for faster convergence.
* **Smart Data Handling**: Automatic Min-Max scaling, shuffling, and 80/20 train/val split.
* **Early Stopping**: Stop wasting CPU cycles once the validation F1-score plateaus.
* **Full Metrics**: Full tracking of Loss, Precision, Recall, and F1-Score.

## Input Data
You can change the number of input taked based from your dataset
1. **Exemple of data.csv for training**
```csv
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001
8610629,B,13.53,10.94,87.91,559.2,0.1291,0.1047
```
2. **Exemple for data.csv for prediction**
```csv
17.99,10.38,122.8,1001,0.1184,0.2776,0.3001
13.53,10.94,87.91,559.2,0.1291,0.1047
```

## Quick Start

1. **Setup the data**:
   Drop your `data.csv` in the root folder, then run:
   ```bash
   cd src
   lua main.lua
   ```

## CLI Arguments

| Flag | Function |
| :--- | :--- |
| `-h, --help` | Display the help message and exit |
| `-r, --reload` | Wipes existing splits and regenerates normalized train/val files from `data.csv` |
| `-t, --train` | Forces a new training session even if a saved model already exists |
| `-n, --normalize` | Manually triggers Min-Max normalization on existing `data_train` and `data_val` |
| `-l, --layer` | Set the number of hidden layers (default: 2) |
| `-ls, --layer-size` | Set the number of neurons per hidden layer (default: 16) |
| `-il, --input-layer` | Set the number of input features (default: 30 for WBCD) |
| `-e, --epoch` | Maximum number of training iterations (default: 200) |
| `-b, --batch` | Set the batch size for the optimizer (default: 32) |
| `-lr, --learning-rate` | Adjust the step size for the Adam optimizer (default: 0.005) |
| `-es, --early-stopping`| Set "patience" (number of epochs to wait without improvement before stopping) |
| `-s, --split` | Set the % of data for the training set (e.g., `0.8` for 80%) |
| `-o, --output` | Specify the filename for the saved model (e.g., `my_model.lua`) |
| `-m, --model` | Choose a specific saved model file to load |
| `-p, --predict` | Use a trained model to run predictions on a specific data file |
| `-np, --normalize-predict` | Normalize the data to predict (do not change the file content) |

## Training Visualization

The project includes a web-based dashboard to monitor your model's metrics (Loss, F1-Score, etc.)

1. **Navigate** to the graph directory:
   ```bash
   cd graph
   ```
2. **Start a local web server** (Python is the easiest way):
    ```python
    python -m http.server
    ```
3. **Open your browser** and go to: `http://localhost:8000`

![Training Curves](graph.png)

## License
This project is licensed under the GNU GPL v3 License - see the [LICENSE](LICENSE) file for details. 
Basically: You can use and modify this, but you must keep it open-source and give credit.
