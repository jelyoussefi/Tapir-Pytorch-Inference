# CenterNet Windows 11 Implementation
# A PowerShell script to replace the Linux/Docker Makefile

# ----------------------------------
# General Settings
# ----------------------------------
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Directories
$DATASET_DIR = Join-Path $SCRIPT_DIR "dataset"
$MODELS_DIR = Join-Path $SCRIPT_DIR "models"
$COCO_DATASET_DIR = Join-Path $SCRIPT_DIR "datasets\COCO"

# Default Parameters
$DEVICE = if ($env:DEVICE) { $env:DEVICE } else { "GPU" }
$INPUT_SIZE = if ($env:INPUT_SIZE) { $env:INPUT_SIZE } else { 640 }
$PRECISION = if ($env:PRECISION) { $env:PRECISION } else { "FP32" }
$CHECKPOINT_PATH = if ($env:CHECKPOINT_PATH) { $env:CHECKPOINT_PATH } else { Join-Path $SCRIPT_DIR "model_epoch_8.pth" }
$CONFIG_PATH = if ($env:CONFIG_PATH) { $env:CONFIG_PATH } else { Join-Path $SCRIPT_DIR "model_centernet_r18_8xb16-crop512-140e_coco.py" }

# Model and Input Paths
$MODEL_PATH = Join-Path $MODELS_DIR "$PRECISION\centernet.xml"
$INPUT_PATH = Join-Path $SCRIPT_DIR "streams\tube_capped.jpg"

# ----------------------------------
# Helper Functions
# ----------------------------------
function Show-Menu {
    Clear-Host
    Write-Host "================ CenterNet OpenVINO on Windows 11 ================"
    Write-Host "1: Run Inference"
    Write-Host "2: Prepare COCO Dataset"
    Write-Host "3: Export PyTorch Model to OpenVINO"
    Write-Host "4: Quantize Model to INT8"
    Write-Host "Q: Quit"
    Write-Host "=============================================================="
}

function Run-Inference {
    Write-Host "🚀 Running CenterNet Inference demo in $PRECISION ..."
    
    # Make sure python is available
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "❌ Error: Python is not installed or not in the PATH" -ForegroundColor Red
        return
    }
    
    # Run the detector script
    python "$SCRIPT_DIR\detector.py" `
        -m $MODEL_PATH `
        -i $INPUT_PATH `
        -d $DEVICE `
        -p $PRECISION
}

function Prepare-Dataset {
    Write-Host "🚀 Preparing the COCO dataset ..."
    
    # Create directory if it doesn't exist
    if (-not (Test-Path $COCO_DATASET_DIR)) {
        New-Item -ItemType Directory -Path $COCO_DATASET_DIR -Force | Out-Null
    }
    
    # Run the dataset preparation script
    # Convert bash script logic to PowerShell
    python "$SCRIPT_DIR\prepare_coco_dataset.py"
}

function Export-ToOpenVINO {
    Write-Host "🚀 Exporting PyTorch model to OpenVINO ..."
    
    # Create output directory if it doesn't exist
    $outputDir = Join-Path $MODELS_DIR "FP32"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Run the export script
    python "$SCRIPT_DIR\openvino_export.py" `
        --config $CONFIG_PATH `
        --checkpoint $CHECKPOINT_PATH `
        --resolution $INPUT_SIZE `
        --output_dir $outputDir
}

function Quantize-Model {
    Write-Host "⚙️ Quantizing model to INT8 ..."
    
    # Make sure the dataset is prepared
    if (-not (Test-Path $COCO_DATASET_DIR)) {
        Write-Host "⚠️ COCO dataset not found. Preparing dataset first..." -ForegroundColor Yellow
        Prepare-Dataset
    }
    
    # Create output directory if it doesn't exist
    $outputDir = Join-Path $MODELS_DIR "INT8"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Run the quantization script
    python "$SCRIPT_DIR\quantize.py" `
        --model (Join-Path $MODELS_DIR "FP32\centernet.xml") `
        --dataset $COCO_DATASET_DIR `
        --resize $INPUT_SIZE $INPUT_SIZE `
        --output (Join-Path $outputDir "centernet.xml")
}

# ----------------------------------
# Main Script
# ----------------------------------

# Check for command line arguments
if ($args.Count -gt 0) {
    switch ($args[0].ToLower()) {
        "run" { Run-Inference; exit }
        "dataset" { Prepare-Dataset; exit }
        "export" { Export-ToOpenVINO; exit }
        "quantize" { Quantize-Model; exit }
        default { 
            Write-Host "Unknown command: $($args[0])"
            Write-Host "Available commands: run, dataset, export, quantize"
            exit 1
        }
    }
}

# If no arguments, show interactive menu
do {
    Show-Menu
    $selection = Read-Host "Please make a selection"
    
    switch ($selection) {
        '1' { Run-Inference; pause }
        '2' { Prepare-Dataset; pause }
        '3' { Export-ToOpenVINO; pause }
        '4' { Quantize-Model; pause }
        'q' { return }
    }
} until ($selection -eq 'q')
